# 文件路径: client/fedsmoo.py

import torch
from .client import Client
from utils import *
from optimizer import *
# 你已有的导入……
from utils import get_mdl_params, param_to_vector
# 将错误的导入语句替换为下面这句
from optimizer.utils_mud import (
    build_flatten_plan,
    stable_layer_seed,
    AAD_decomposition,
    post_hoc_decomposition,
    generate_aad_basis,
    bkd_decomposition,
    bkd_recover,
    bkd_aad_recover,
    allocate_budget,
)


class fedsmoo(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedsmoo, self).__init__(device, model_func, received_vecs, dataset, lr, args)

        # FedSMOO 组件（保持你原逻辑）
        self.base_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.args.weight_decay + self.args.lamb,
        )
        self.optimizer = DRegSAM(self.model.parameters(), self.base_optimizer, rho=self.args.rho)

        # MUD 压缩配置
        self.use_mud = bool(getattr(args, 'use_mud', False))
        self.rank = int(getattr(args, 'rank', getattr(args, 'mud_rank', 0)))
        self.use_aad = bool(getattr(args, 'use_aad', False))
        self.enable_bkd = bool(getattr(args, 'enable_bkd', False))
        self.target_cr = float(getattr(args, 'target_cr', 0.0))
        self.kron_blocks = int(getattr(args, 'kron_blocks', 1))
        self.aad_seed = int(getattr(args, 'aad_seed', 0))
        self.aad_pattern = getattr(args, 'aad_pattern', 'none').lower()

        # 冷启动与频率控制
        self.warmup_rounds = int(args.warmup_rounds)  # 前 N 轮不压缩
        self.compress_every = int(args.compress_every)  # 每 K 轮压一次
        self.skip_threshold = float(args.skip_threshold)  # 跳过阈值

        # 构建/复用展平计划（严格按 named_parameters 顺序）
        if self.use_mud and self.received_vecs.get('flatten_plan'):
            # 复用服务器下发的展平计划，避免重复构建
            self.flatten_plan = [dict(item) for item in self.received_vecs['flatten_plan']]
            total = self.received_vecs.get('total_params')
            if total is None and self.flatten_plan:
                total = self.flatten_plan[-1]['end']
            self.total_params = total if total is not None else sum(
                item['end'] - item['start'] for item in self.flatten_plan)
        else:
            self.flatten_plan, self.total_params = build_flatten_plan(self.model)

        self.ef_memory = {}
        if self.use_mud:
            param_dtype = next(self.model.parameters()).dtype
            for item in self.flatten_plan:
                if item.get('is_target'):
                    self.ef_memory[item['name']] = torch.zeros(item['shape'], dtype=param_dtype)

        self.layer_budget = {}
        if self.use_mud and self.target_cr > 0:
            mode = self.aad_pattern if self.enable_bkd else ('aad' if self.use_aad else 'svd')
            self.layer_budget = allocate_budget(self.flatten_plan, self.target_cr, mode=mode,
                                                base_floor=max(1, self.rank))

    # =====================================================================
    #  ↓↓↓ 这里是修改的核心区域 ↓↓↓
    # =====================================================================
    def train(self):
        # local training
        self.model.train()

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                # 定义一个临时的、包含所有损失项的函数
                # 这是为了让 DRegSAM 能够对完整的总损失进行 SAM 优化
                def combined_loss_func(model_predictions, ground_truth_labels):
                    # 1. 计算基础的交叉熵损失
                    loss_pred = self.loss(model_predictions, ground_truth_labels)

                    # 2. 计算动态正则化项
                    param_vec = param_to_vector(self.model)  # 获取当前最新的模型参数
                    delta_vec = self.received_vecs['Local_dual_correction'].to(self.device)
                    loss_correct = self.args.lamb * torch.sum(param_vec * delta_vec)

                    # 3. 返回总损失
                    return loss_pred + loss_correct

                # 将输入、标签和我们新定义的组合损失函数传递给 DRegSAM 优化器
                self.optimizer.paras = [inputs, labels, combined_loss_func, self.model,
                                        self.received_vecs['Dynamic_dual_correction']]

                # 执行 DRegSAM 的完整一步优化
                # 它内部会处理两次反向传播和梯度扰动
                self.received_vecs['Dynamic_dual'] = self.optimizer.step(self.received_vecs['Dynamic_dual'])

                # 在 DRegSAM 完成其内部的梯度计算后，使用基础优化器（SGD）来应用这个最终的梯度
                # 这一步是必须的，因为它完成了权重的最终更新
                self.base_optimizer.step()

        # =====================================================================
        #  ↑↑↑ 修改结束 ↑↑↑
        # =====================================================================

        # === 通信阶段 (这部分保持你原来的逻辑不变) ===
        last_state_params = get_mdl_params(self.model, device=torch.device("cpu"))
        base_vec = self.received_vecs['Params_list'].to(last_state_params.device)
        delta_w_full = last_state_params - base_vec

        current_round = int(self.received_vecs.get('round', 0))
        should_compress = (
                self.use_mud
                and current_round >= self.warmup_rounds
                and (self.compress_every <= 1 or current_round % self.compress_every == 0)
        )

        comm_vecs = {
            'use_compression': False,
            'compressed_update': None,
            'local_update_list': None,
            'local_dynamic_dual': self.received_vecs['Dynamic_dual'],
            'aad_seed': self.received_vecs.get('aad_seed', self.aad_seed) if self.use_aad else None,
            'round': current_round,
        }

        if should_compress:
            compressed = self.compress_update_post_hoc(delta_w_full)
            if compressed:
                comm_vecs['use_compression'] = True
                comm_vecs['compressed_update'] = compressed
            else:
                comm_vecs['local_update_list'] = delta_w_full
        else:
            comm_vecs['local_update_list'] = delta_w_full

        return comm_vecs

    def compress_update_post_hoc(self, delta_w):
        """
        后置压缩（更省带宽版本）：
        - 仅对目标层进行压缩
        - 微小更新层直接跳过（不发占位包）
        - 统一 CPU 传输
        """
        try:
            compressed = {}
            skipped, done = 0, 0
            layer_cache = {}
            importance = {}

            # 第一遍：缓存张量和重要性度量
            for item in self.flatten_plan:
                layer_delta = delta_w[item['start']:item['end']].reshape(item['shape'])
                orig_shape = tuple(layer_delta.shape)

                if item['is_target']:
                    ef_tensor = self.ef_memory.get(item['name'])
                    if ef_tensor is None or ef_tensor.shape != orig_shape:
                        ef_tensor = torch.zeros_like(layer_delta)
                    delta_with_ef = layer_delta + ef_tensor.to(layer_delta.dtype)
                    layer_cache[item['name']] = {
                        'delta': layer_delta,
                        'delta_with_ef': delta_with_ef,
                        'orig_shape': orig_shape,
                    }
                    if self.target_cr > 0:
                        importance[item['name']] = torch.norm(delta_with_ef).item()
                else:
                    layer_cache[item['name']] = {
                        'delta': layer_delta,
                        'orig_shape': orig_shape,
                    }

            # 自适应预算：叠加最新 ΔW 的范数
            if self.use_mud and self.target_cr > 0:
                mode = self.aad_pattern if self.enable_bkd else ('aad' if self.use_aad else 'svd')
                dynamic_budget = allocate_budget(
                    self.flatten_plan,
                    self.target_cr,
                    mode=mode,
                    base_floor=max(1, self.rank),
                    importance=importance,
                )
            else:
                dynamic_budget = self.layer_budget

            if dynamic_budget:
                self.layer_budget = dynamic_budget

            # 第二遍：执行实际压缩
            for item in self.flatten_plan:
                info = layer_cache[item['name']]
                orig_shape = info['orig_shape']

                if item['is_target']:
                    delta_with_ef = info['delta_with_ef']
                    current_norm = torch.norm(delta_with_ef).item()

                    if current_norm < self.skip_threshold:
                        compressed[item['name']] = {
                            'type': 'dense',
                            'shape': orig_shape,
                            'tensor': delta_with_ef.detach().cpu(),
                        }
                        self.ef_memory[item['name']] = torch.zeros_like(delta_with_ef).detach().cpu()
                        skipped += 1
                        continue

                    cfg = dynamic_budget.get(item['name'], {}) if dynamic_budget else {}
                    rank = int(cfg.get('rank', self.rank if self.rank > 0 else 1))
                    blocks = int(cfg.get('blocks', self.kron_blocks if self.kron_blocks > 0 else 1))
                    rank = max(rank, 1)
                    blocks = max(blocks, 1)

                    if len(orig_shape) > 2:
                        delta_2d = delta_with_ef.reshape(orig_shape[0], -1)
                    else:
                        delta_2d = delta_with_ef

                    if self.enable_bkd:
                        U_list, V_list, meta = bkd_decomposition(delta_2d, rank=rank, blocks=blocks)
                        aad_seed = self.received_vecs.get('aad_seed', self.aad_seed)
                        if self.use_aad:
                            approx_2d = bkd_aad_recover(
                                U_list,
                                V_list,
                                meta,
                                aad_seed=aad_seed,
                                layer_name=item['name'],
                                device=delta_2d.device,
                                dtype=delta_2d.dtype,
                            )
                            pack_type = 'bkd_aad'
                        else:
                            approx_2d = bkd_recover(
                                U_list,
                                V_list,
                                meta,
                                device=delta_2d.device,
                                dtype=delta_2d.dtype,
                            )
                            pack_type = 'bkd'

                        approx_layer = (
                            approx_2d.reshape(orig_shape)
                            if len(orig_shape) > 2
                            else approx_2d
                        )
                        self.ef_memory[item['name']] = (delta_with_ef - approx_layer).detach().cpu()
                        compressed[item['name']] = {
                            'type': pack_type,
                            'shape': orig_shape,
                            'U_list': [U.cpu() for U in U_list],
                            'V_list': [V.cpu() for V in V_list],
                            'meta': meta,
                        }
                    else:
                        aad_seed = self.received_vecs.get('aad_seed', self.aad_seed)
                        if self.use_aad:
                            layer_seed = stable_layer_seed(aad_seed, item['name'])
                            U, V = AAD_decomposition(delta_2d, rank, layer_seed, lambda_reg=self.args.als_reg)
                            U_tilde, V_tilde = generate_aad_basis(
                                seed=layer_seed,
                                m=U.shape[0],
                                n=V.shape[0],
                                rank=U.shape[1],
                                device=U.device,
                            )
                            approx_2d = U @ V_tilde.T + U_tilde @ V.T
                            pack_type = 'aad'
                        else:
                            U, V = post_hoc_decomposition(delta_2d, rank)
                            approx_2d = U @ V.T
                            pack_type = 'svd'

                        approx_layer = (
                            approx_2d.reshape(orig_shape)
                            if len(orig_shape) > 2
                            else approx_2d
                        )
                        self.ef_memory[item['name']] = (delta_with_ef - approx_layer).detach().cpu()
                        compressed[item['name']] = {
                            'type': pack_type,
                            'shape': orig_shape,
                            'U': U.detach().cpu(),
                            'V': V.detach().cpu(),
                        }

                    done += 1
                else:
                    compressed[item['name']] = {
                        'type': 'dense',
                        'shape': orig_shape,
                        'tensor': info['delta'].detach().cpu(),
                    }

            if getattr(self.args, "verbose", False):
                print(f"[MUD] compressed={done}, skipped={skipped}")

            return compressed if compressed else None

        except Exception as e:
            print(f"[MUD] Compression failed: {e}")
            return None
