import os
import time
import datetime
from typing import Dict, Optional

import numpy as np
import torch
from utils import *
from utils.comm_meter import CommMeter
from utils.comm_size import count_tensor_bytes, dense_bytes_from_shapes
from dataset import Dataset
from torch.utils import data

from optimizer.utils_mud import (
    build_flatten_plan,
    stable_layer_seed,
    generate_aad_basis,
    bkd_recover,
    bkd_aad_recover,
    AAD_decomposition,
    post_hoc_decomposition,
    bkd_decomposition,
)


COMM_METER = None


def init_comm_meter(args):
    global COMM_METER
    baseline = getattr(args, 'comm_baseline', 'dense_delta')
    log_dir = getattr(args, 'log_dir', getattr(args, 'out_file', './out'))
    args.comm_baseline = baseline
    args.log_dir = log_dir
    COMM_METER = CommMeter(log_dir, baseline=baseline)


class Server(object):
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args):
        self.args = args
        if not getattr(self.args, 'log_dir', None):
            self.args.log_dir = getattr(self.args, 'out_file', './out')
        if not getattr(self.args, 'comm_baseline', None):
            self.args.comm_baseline = 'dense_delta'
        init_comm_meter(self.args)
        self.device = device
        self.datasets = datasets
        self.model_func = model_func

        self.server_model = init_model.to(self.device)
        self.server_model_params_list = init_par_list

        print("Initialize the Server      --->  {:s}".format(self.args.method))
        print("Initialize the Public Storage:")
        self.clients_params_list = init_par_list.repeat(args.total_client, 1)
        print("   Local Model Param List  --->  {:d} * {:d}".format(
            self.clients_params_list.shape[0], self.clients_params_list.shape[1]))

        self.clients_updated_params_list = torch.zeros((args.total_client, init_par_list.shape[0]))
        print(" Local Updated Param List  --->  {:d} * {:d}".format(
            self.clients_updated_params_list.shape[0], self.clients_updated_params_list.shape[1]))

        self.train_perf = np.zeros((self.args.comm_rounds, 2))
        self.test_perf = np.zeros((self.args.comm_rounds, 2))
        print("   Train/Test [loss, acc]  --->  {:d} * {:d}".format(self.train_perf.shape[0], self.train_perf.shape[1]))
        self.divergence = np.zeros((args.comm_rounds))
        print("  Consistency (Divergence) --->  {:d}".format(self.divergence.shape[0]))

        self.time = np.zeros((args.comm_rounds))
        self.lr = self.args.local_learning_rate

        self.comm_vecs = {'Params_list': None}
        self.received_vecs = None
        self.Client = None

        # =====================================================================
        #  ↓↓↓ 改动 1: 在 __init__ 中添加 MUD/AAD 相关配置 ↓↓↓
        # =====================================================================
        if self.args.use_mud:
            print("MUD/AAD Compression is Enabled.")
            self.flatten_plan, self.total_params = build_flatten_plan(self.server_model)
        else:
            self.flatten_plan = None
            self.total_params = init_par_list.shape[0]

        self._target_layer_shapes = None
        if self.flatten_plan is not None:
            targets = [meta['shape'] for meta in self.flatten_plan if meta.get('is_target', False)]
            if targets:
                self._target_layer_shapes = targets
            else:
                self._target_layer_shapes = [meta['shape'] for meta in self.flatten_plan]
        self._full_model_shapes = [
            p.shape for p in self.server_model.state_dict().values()
            if torch.is_tensor(p) and p.dtype.is_floating_point
        ]

        self.latest_downlink_payload = None
        self.latest_dmu_seed_map = None

        if getattr(self.args, 'use_aad', False):
            self.aad_seed = self.args.aad_seed
        # =====================================================================

    def _baseline_bytes(self):
        baseline = getattr(self.args, 'comm_baseline', 'dense_delta')
        if baseline == 'full_model':
            shapes = self._full_model_shapes
        else:
            shapes = self._target_layer_shapes or self._full_model_shapes
        return dense_bytes_from_shapes(shapes, dtype=torch.float32) if shapes else 0

    @staticmethod
    def _infer_payload_mode(payload):
        if isinstance(payload, dict):
            modes = {
                entry.get('type', 'unknown')
                for entry in payload.values()
                if isinstance(entry, dict) and entry.get('type')
            }
            if len(modes) == 1:
                return next(iter(modes))
            if len(modes) > 1:
                return '+'.join(sorted(modes))
        return 'unknown'

    # =====================================================================
    #  ↓↓↓ 改动 2: 添加用于解压/重建的辅助函数 ↓↓↓
    # =====================================================================
    def reconstruct_delta_vec(self, compressed: dict):
        """
        从客户端发来的压缩包中重建完整的模型更新向量 (ΔW)。
        """
        out = torch.zeros_like(self.server_model_params_list, device='cpu')

        for item in self.flatten_plan:
            pack = compressed.get(item['name'])
            if pack is None:
                continue

            if pack['type'] in {'dense', 'dense_residual'}:
                layer_delta = pack['tensor'].to(out.dtype)
                full_layer_delta = layer_delta.reshape(pack['shape']).reshape(-1)
            else:
                pack_type = pack['type']
                if pack_type in {'svd', 'aad'}:
                    U = pack['U']
                    V = pack['V']

                    if pack_type == 'svd':
                        M = U @ V.T
                    else:
                        m, n = U.shape[0], V.shape[0]
                        rank = U.shape[1]
                        layer_seed = stable_layer_seed(self.aad_seed, item['name'])
                        U_tilde, V_tilde = generate_aad_basis(seed=layer_seed, m=m, n=n, rank=rank, device='cpu')
                        M = U @ V_tilde.T + U_tilde @ V.T
                elif pack_type == 'bkd':
                    M = bkd_recover(pack['U_list'], pack['V_list'], pack['meta'])
                elif pack_type == 'bkd_aad':
                    M = bkd_aad_recover(
                        pack['U_list'],
                        pack['V_list'],
                        pack['meta'],
                        aad_seed=self.aad_seed,
                        layer_name=item['name'],
                    )
                else:
                    continue

                target_shape = pack.get('shape')
                if target_shape is None and 'meta' in pack:
                    target_shape = pack['meta']['shape']
                if target_shape is None:
                    target_shape = M.shape
                full_layer_delta = M.reshape(target_shape).reshape(-1)

            out[item['start']:item['end']] = full_layer_delta

        return out.to(self.server_model_params_list.device, dtype=self.server_model_params_list.dtype)

    def pack_global_update(self, delta_vec: torch.Tensor) -> Optional[Dict[str, Dict[str, object]]]:
        """Pack a dense global update vector for downlink broadcasting."""

        if not self.args.use_mud or self.flatten_plan is None:
            return None

        packed: Dict[str, Dict[str, object]] = {}
        rank = max(1, int(getattr(self.args, 'rank', 1)))
        lambda_reg = float(getattr(self.args, 'als_reg', 1e-4))
        use_aad = bool(getattr(self.args, 'use_aad', False))

        for item in self.flatten_plan:
            start, end = item['start'], item['end']
            if start >= end:
                continue

            layer_delta = delta_vec[start:end].reshape(item['shape'])
            entry: Dict[str, object] = {'shape': item['shape']}

            if not item.get('is_target', False):
                entry.update({'type': 'dense', 'tensor': layer_delta.detach().cpu()})
                packed[item['name']] = entry
                continue

            matrix = layer_delta.reshape(layer_delta.shape[0], -1) if layer_delta.ndim > 2 else layer_delta
            matrix = matrix.to(dtype=torch.float32)

            if use_aad:
                layer_seed = stable_layer_seed(self.aad_seed, item['name'])
                U, V = AAD_decomposition(matrix, rank, layer_seed, lambda_reg=lambda_reg)
                entry.update({
                    'type': 'aad',
                    'U': U.detach().cpu(),
                    'V': V.detach().cpu(),
                })
            else:
                U, V = post_hoc_decomposition(matrix, rank)
                entry.update({
                    'type': 'svd',
                    'U': U.detach().cpu(),
                    'V': V.detach().cpu(),
                })

            packed[item['name']] = entry

        return packed if packed else None

    def _build_dmu_seed_map(self, round_idx: int) -> Optional[Dict[str, int]]:
        if not getattr(self.args, 'use_dmu', False) or self.flatten_plan is None:
            return None

        base_seed = int(getattr(self.args, 'dmu_seed', getattr(self.args, 'aad_seed', 0)))
        seeds: Dict[str, int] = {}
        for item in self.flatten_plan:
            seeds[item['name']] = int(stable_layer_seed(base_seed + round_idx, item['name']))

        return seeds if seeds else None

    # =====================================================================

    def _see_the_divergence_(self, selected_clients, t):
        self.divergence[t] = torch.norm(
            self.clients_params_list[selected_clients] - self.server_model_params_list) ** 2 / len(selected_clients)

    def _activate_clients_(self, t):
        inc_seed = 0
        while (True):
            np.random.seed(t + self.args.seed + inc_seed)
            act_list = np.random.uniform(size=self.args.total_client)
            act_clients = act_list <= self.args.active_ratio
            selected_clients = np.sort(np.where(act_clients)[0])
            inc_seed += 1
            if len(selected_clients) != 0:
                return selected_clients

    def _lr_scheduler_(self):
        self.lr *= self.args.lr_decay

    def _test_(self, t, selected_clients):
        loss, acc = self._validate_(
            (np.concatenate(self.datasets.client_x, axis=0), np.concatenate(self.datasets.client_y, axis=0)))
        self.train_perf[t] = [loss, acc]
        print(
            "   Train    ----    Loss: {:.4f},   Accuracy: {:.4f}".format(self.train_perf[t][0], self.train_perf[t][1]),
            flush=True)
        loss, acc = self._validate_((self.datasets.test_x, self.datasets.test_y))
        self.test_perf[t] = [loss, acc]
        print("    Test    ----    Loss: {:.4f},   Accuracy: {:.4f}".format(self.test_perf[t][0], self.test_perf[t][1]),
              flush=True)
        self._see_the_divergence_(selected_clients, t)
        print("            ----    Divergence: {:.4f}".format(self.divergence[t]), flush=True)

    def _summary_(self):
        print("##=============================================##")
        print("##                   Summary                   ##")
        print("##=============================================##")
        print("     Communication round   --->   T = {:d}       ".format(self.args.comm_rounds))
        print("    Average Time / round   --->   {:.2f}s        ".format(np.mean(self.time)))
        print("     Top-1 Test Acc (T)    --->   {:.2f}% ({:d}) ".format(np.max(self.test_perf[:, 1]) * 100.,
                                                                         np.argmax(self.test_perf[:, 1])))

    def _validate_(self, dataset):
        self.server_model.eval()
        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        testdataset = data.DataLoader(Dataset(dataset[0], dataset[1], train=False, dataset_name=self.args.dataset),
                                      batch_size=1000, shuffle=False)

        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testdataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                predictions = self.server_model(inputs)
                loss = self.loss(predictions, labels)
                total_loss += loss.item()

                predictions = predictions.cpu().numpy()
                predictions = np.argmax(predictions, axis=1).reshape(-1)
                labels = labels.cpu().numpy().reshape(-1).astype(np.int32)
                batch_correct = np.sum(predictions == labels)
                total_acc += batch_correct

        # ... (前面的循环代码) ...

        if self.args.weight_decay != 0.:
            # Add L2 loss
            # 计算权重衰减项，它是一个 Tensor
            weight_decay_loss = self.args.weight_decay / 2. * torch.sum(
                self.server_model_params_list * self.server_model_params_list)

            # 使用 .item() 将其转换为 Python float，然后与 total_loss 相加
            total_loss += weight_decay_loss.item()

        return total_loss / (i + 1), total_acc / dataset[0].shape[0]

    def _save_results_(self):
        options = ''
        root = '{:s}/T={:d}'.format(self.args.out_file, self.args.comm_rounds)
        if not os.path.exists(root):
            os.makedirs(root)
        if not self.args.non_iid:
            root += '/{:s}-{:s}{:s}-{:d}'.format(self.args.dataset, 'IID', '', self.args.total_client)
        else:
            root += '/{:s}-{:s}{:s}-{:d}'.format(self.args.dataset, self.args.split_rule, str(self.args.split_coef),
                                                 self.args.total_client)
        if not os.path.exists(root):
            os.makedirs(root)

        participation = str(self.args.active_ratio)
        root = root + '/active-' + participation

        if not os.path.exists(root):
            os.makedirs(root)

        perf_dir = root + '/Performance'
        if not os.path.exists(perf_dir):
            os.makedirs(perf_dir)
        train_file = perf_dir + '/trn-{:s}{:s}.npy'.format(self.args.method, options)
        test_file = perf_dir + '/tst-{:s}{:s}.npy'.format(self.args.method, options)
        np.save(train_file, self.train_perf)
        np.save(test_file, self.test_perf)

        divergence_dir = root + '/Divergence'
        if not os.path.exists(divergence_dir):
            os.makedirs(divergence_dir)
        divergence_file = divergence_dir + '/divergence-{:s}{:s}.npy'.format(self.args.method, options)
        np.save(divergence_file, self.divergence)

    def process_for_communication(self, client, Averaged_update):
        pass

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        pass

    def postprocess(self, client, received_vecs):
        pass

    def train(self):
        print("##=============================================##")
        print("##           Training Process Starts           ##")
        print("##=============================================##")

        Averaged_update = torch.zeros(self.server_model_params_list.shape)

        for t in range(self.args.comm_rounds):
            self.current_round = t
            start = time.time()
            selected_clients = self._activate_clients_(t)
            print('============= Communication Round', t + 1, '=============', flush=True)
            print('Selected Clients: %s' % (', '.join(['%2d' % item for item in selected_clients])))

            for client in selected_clients:
                dataset = (self.datasets.client_x[client], self.datasets.client_y[client])
                self.process_for_communication(client, Averaged_update)

                if self.args.use_mud:
                    self.comm_vecs['down_payload_t'] = self.latest_downlink_payload
                    self.comm_vecs['dmu_seed_map'] = self.latest_dmu_seed_map
                    self.comm_vecs['apply_global_update'] = bool(self.latest_downlink_payload)
                else:
                    self.comm_vecs.pop('down_payload_t', None)
                    self.comm_vecs.pop('dmu_seed_map', None)
                    self.comm_vecs.pop('apply_global_update', None)

                self.comm_vecs['round'] = int(getattr(self, 'current_round', 0))
                if getattr(self.args, 'use_aad', False):
                    self.comm_vecs['aad_seed'] = self.aad_seed

                if COMM_METER is not None:
                    payload_for_client = self.comm_vecs.get('down_payload_t')
                    actual = count_tensor_bytes(payload_for_client) if payload_for_client else 0
                    baseline = self._baseline_bytes()
                    mode = self._infer_payload_mode(payload_for_client) if payload_for_client else 'dense'
                    COMM_METER.add('downlink', actual, baseline, mode)

                _edge_device = self.Client(device=self.device, model_func=self.model_func, received_vecs=self.comm_vecs,
                                           dataset=dataset, lr=self.lr, args=self.args)

                rec = _edge_device.train()

                if COMM_METER is not None and isinstance(rec, dict):
                    metrics = rec.get('metrics') or {}
                    up_act = int(metrics.get('uplink_actual_bytes', 0))
                    up_base = int(metrics.get('uplink_baseline_bytes', 0))
                    up_mode = metrics.get('uplink_mode', 'unknown')
                    if up_act or up_base:
                        COMM_METER.add('uplink', up_act, up_base, up_mode)

                # =====================================================================
                #  ↓↓↓ 改动 3: 替换原有的接收逻辑，智能处理压缩/非压缩更新 ↓↓↓
                # =====================================================================
                # 步骤 A: 确定本轮客户端的完整模型更新 ΔW (delta_vec)
                if rec.get('use_compression', False) and rec.get('compressed_update') is not None:
                    # 如果是压缩模式，则从压缩包重建 ΔW
                    delta_vec = self.reconstruct_delta_vec(rec['compressed_update'])
                else:
                    # 否则，直接使用客户端返回的全量 ΔW
                    delta_vec = rec['local_update_list']

                # 将计算好的 ΔW 存入服务器记录，用于后续聚合
                self.clients_updated_params_list[client] = delta_vec

                # 步骤 B: 确定客户端训练结束时的模型参数 W_t+1
                if 'local_model_param_list' in rec and rec['local_model_param_list'] is not None:
                    # 如果客户端返回了完整的模型参数，直接使用
                    self.clients_params_list[client] = rec['local_model_param_list']
                else:
                    # 如果没返回（压缩模式），则通过 W_t + ΔW 在服务器端推算出来
                    # 用“本轮下发给该客户端的起点”来回推（即便启用了 RI 也没问题）
                    self.clients_params_list[client] = self.comm_vecs['Params_list'] + delta_vec
                # =====================================================================

                self.postprocess(client, rec)

                del _edge_device

            Averaged_update = torch.mean(self.clients_updated_params_list[selected_clients], dim=0)
            Averaged_model = torch.mean(self.clients_params_list[selected_clients], dim=0)

            prev_global_params = self.server_model_params_list.clone()
            self.server_model_params_list = self.global_update(selected_clients, Averaged_update, Averaged_model)
            set_mdl_params(self.server_model, self.server_model_params_list)

            if self.args.use_mud:
                delta_global = (self.server_model_params_list - prev_global_params).detach().cpu()
                self.latest_downlink_payload = self.pack_global_update(delta_global)
                round_idx = int(getattr(self, 'current_round', 0)) + 1
                self.latest_dmu_seed_map = self._build_dmu_seed_map(round_idx)
            else:
                self.latest_downlink_payload = None
                self.latest_dmu_seed_map = None

            self._test_(t, selected_clients)
            self._lr_scheduler_()

            end = time.time()
            self.time[t] = end - start
            print("            ----    Time: {:.2f}s".format(self.time[t]), flush=True)

            if COMM_METER is not None:
                COMM_METER.tick_round()

        if COMM_METER is not None:
            summary = COMM_METER.summary()
            COMM_METER.dump(filename='comm_report.json')
            print(
                "[Comm] totalCR={:.2f}x, uplinkCR={:.2f}x, downlinkCR={:.2f}x, totalMB={:.2f}/{:.2f} (actual/baseline)".format(
                    summary['total_cr'], summary['uplink_cr'], summary['downlink_cr'],
                    summary['total_MB'], summary['total_baseline_MB']
                )
            )

        self._save_results_()
        self._summary_()
