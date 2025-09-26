from typing import Dict, Optional

import torch
from utils import *
from dataset import Dataset
from torch.utils import data
# from utils_optimizer import *
from nn import DMU_Linear, DMU_Conv2d
from nn.dmu import _reset_update_inplace
from optimizer.utils_mud import (
    stable_layer_seed,
    generate_aad_basis,
    bkd_recover,
    bkd_aad_recover,
)


class Client():
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        self.args = args
        self.device = device
        self.model_func = model_func
        self.received_vecs = received_vecs
        self.comm_vecs = {
            'local_update_list': None,
            'local_model_param_list': None,
        }
        
        self.use_dmu = bool(getattr(self.args, 'use_dmu', False))
        params_vec = self.received_vecs.get('Params_list')
        if params_vec is None and not (
            self.use_dmu and self.received_vecs.get('apply_global_update', False)
        ):
            raise Exception("CommError: invalid vectors Params_list received")
        self.model = self.model_func().to(self.device)
        if params_vec is not None:
            set_mdl_params(self.model, params_vec)

        if self.use_dmu:
            pack = self.received_vecs.get('down_payload_t')
            if pack is None:
                pack = self.received_vecs.get('global_update_pack')
            seed_map = self.received_vecs.get('dmu_seed_map')
            apply_flag = bool(self.received_vecs.get('apply_global_update', False))
            self._apply_downlink_payload(pack, seed_map, apply_flag)
            if apply_flag:
                self.received_vecs['Params_list'] = get_mdl_params(self.model, device=torch.device('cpu'))

        self.loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        self.dataset = data.DataLoader(Dataset(dataset[0], dataset[1], train=True, dataset_name=self.args.dataset), batch_size=self.args.batchsize, shuffle=True)
        
        self.max_norm = 10

    def _apply_downlink_payload(self, pack: Optional[Dict[str, Dict]], seed_map: Optional[Dict[str, int]],
                                apply_flag: bool) -> None:
        if not apply_flag or not pack:
            if apply_flag:
                # Even if pack is missing, keep deterministic seeds in sync.
                self._reset_all_dmu(seed_map)
            return

        init_mag = float(getattr(self.args, 'dmu_init_mag', 1e-3))
        base_seed = int(getattr(self.args, 'dmu_seed', getattr(self.args, 'aad_seed', 0)))
        round_idx = int(self.received_vecs.get('round', 0)) + 1
        aad_seed = int(self.received_vecs.get('aad_seed', getattr(self.args, 'aad_seed', 0)))

        for module_name, module in self.model.named_modules():
            if not isinstance(module, (DMU_Linear, DMU_Conv2d)):
                continue

            weight_name = f"{module_name}.weight" if module_name else "weight"
            entry = pack.get(weight_name)
            seed = None
            if seed_map is not None:
                seed = seed_map.get(weight_name)
            if seed is None:
                seed = int(stable_layer_seed(base_seed + round_idx, weight_name))

            if entry is None:
                _reset_update_inplace(module.update, seed, init_mag)
                continue

            pack_type = entry.get('type', 'dense')
            if pack_type == 'aad':
                U = entry['U'].to(device=module.update.U.device, dtype=module.update.U.dtype)
                V = entry['V'].to(device=module.update.V.device, dtype=module.update.V.dtype)
                module.update.U.copy_(U)
                module.update.V.copy_(V)
                module.push_reset_update(seed=seed, init_mag=init_mag)
            else:
                dense = self._reconstruct_dense_update(entry, weight_name, aad_seed)
                dense = dense.to(device=module.weight.device, dtype=module.weight.dtype)
                module.weight.add_(dense)
                _reset_update_inplace(module.update, seed, init_mag)

    def _reset_all_dmu(self, seed_map: Optional[Dict[str, int]]) -> None:
        init_mag = float(getattr(self.args, 'dmu_init_mag', 1e-3))
        base_seed = int(getattr(self.args, 'dmu_seed', getattr(self.args, 'aad_seed', 0)))
        round_idx = int(self.received_vecs.get('round', 0)) + 1

        for module_name, module in self.model.named_modules():
            if not isinstance(module, (DMU_Linear, DMU_Conv2d)):
                continue

            weight_name = f"{module_name}.weight" if module_name else "weight"
            seed = None
            if seed_map is not None:
                seed = seed_map.get(weight_name)
            if seed is None:
                seed = int(stable_layer_seed(base_seed + round_idx, weight_name))
            _reset_update_inplace(module.update, seed, init_mag)

    def _reconstruct_dense_update(self, entry: Dict, layer_name: str, aad_seed: int) -> torch.Tensor:
        pack_type = entry.get('type', 'dense')
        shape = tuple(entry.get('shape', ()))

        if pack_type == 'dense':
            tensor = entry['tensor']
            return tensor.reshape(shape) if shape else tensor

        if pack_type == 'svd':
            U = entry['U']
            V = entry['V']
            return (U @ V.T).reshape(shape)

        if pack_type == 'aad':
            U = entry['U']
            V = entry['V']
            rank = U.shape[1]
            m, n = U.shape[0], V.shape[0]
            layer_seed = stable_layer_seed(aad_seed, layer_name)
            U_tilde, V_tilde = generate_aad_basis(layer_seed, m, n, rank, device=U.device)
            return (U @ V_tilde.T + U_tilde @ V.T).reshape(shape)

        if pack_type == 'bkd':
            return bkd_recover(entry['U_list'], entry['V_list'], entry['meta']).reshape(shape)

        if pack_type == 'bkd_aad':
            return bkd_aad_recover(
                entry['U_list'],
                entry['V_list'],
                entry['meta'],
                aad_seed=aad_seed,
                layer_name=layer_name,
            ).reshape(shape)

        raise ValueError(f"Unsupported pack type: {pack_type}")

    def train(self):
        # local training
        self.model.train()

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                predictions = self.model(inputs)
                loss = self.loss(predictions, labels)

                self.optimizer.zero_grad()
                loss.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)
                self.optimizer.step()

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs