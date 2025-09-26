import math
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import hashlib


def stable_layer_seed(global_seed: int, name: str) -> int:
    """
    生成稳定的层级种子（SHA1 + LCG）
    跨进程、跨平台一致
    """
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    return (global_seed * 1664525 + h) & 0x7fffffff


def build_flatten_plan(model):
    """
    构建参数展平计划
    严格按照named_parameters()顺序，与parameter_to_vector对齐
    """
    plan = []
    current_offset = 0

    for name, param in model.named_parameters():
        param_size = param.numel()

        # 判断是否为压缩目标（Conv/Linear的weight）
        is_target = False
        if name.endswith('.weight'):
            module_name = '.'.join(name.split('.')[:-1])
            try:
                module = model
                for attr in module_name.split('.'):
                    module = getattr(module, attr)
                is_target = isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
            except:
                is_target = False

        plan.append({
            'name': name,
            'shape': tuple(param.shape),
            'start': current_offset,
            'end': current_offset + param_size,
            'is_target': is_target
        })
        current_offset += param_size

    return plan, current_offset


def generate_aad_basis(seed, m, n, rank, device="cpu"):
    """
    生成AAD随机正交基
    使用本地Generator，完全不影响全局RNG
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    # CPU生成确保一致性
    U_tilde = torch.randn(m, rank, generator=g)
    V_tilde = torch.randn(n, rank, generator=g)

    # QR分解获得正交基
    U_tilde, _ = torch.linalg.qr(U_tilde, mode='reduced')
    V_tilde, _ = torch.linalg.qr(V_tilde, mode='reduced')

    # 转到目标设备
    return U_tilde.to(device, torch.float32), V_tilde.to(device, torch.float32)


def _derive_block_seed(global_seed: int, layer_name: str, bi: int, bj: int) -> int:
    """Derive a deterministic seed for a specific (bi, bj) block."""

    layer_token = f"{layer_name}|{bi}|{bj}"
    return stable_layer_seed(global_seed, layer_token)


def bkd_decomposition(delta_w: torch.Tensor, rank: int, blocks: int):
    """
    Block-wise Kronecker Decomposition (BKD).

    Args:
        delta_w: Tensor of shape (m, n) or higher-rank tensor (will be flattened).
        rank: Target rank per layer (split across blocks).
        blocks: Number of blocks per dimension (k).

    Returns:
        (U_list, V_list, meta) where U_list/V_list contain factors for each block.
    """

    orig_shape = delta_w.shape
    if len(orig_shape) > 2:
        delta_w = delta_w.reshape(orig_shape[0], -1)

    m, n = delta_w.shape
    k = max(1, int(blocks))

    m_pad = (m + k - 1) // k * k
    n_pad = (n + k - 1) // k * k

    pad = torch.zeros((m_pad, n_pad), device=delta_w.device, dtype=delta_w.dtype)
    pad[:m, :n] = delta_w

    mh, nh = m_pad // k, n_pad // k

    U_list: List[torch.Tensor] = []
    V_list: List[torch.Tensor] = []

    # 简化版：每块做 SVD 取 rank_b=r（可线性随块缩放）
    r_b = max(1, int(rank) // k)
    for bi in range(k):
        for bj in range(k):
            block = pad[bi * mh:(bi + 1) * mh, bj * nh:(bj + 1) * nh]
            U, S, Vh = torch.linalg.svd(block, full_matrices=False)
            r_eff = min(r_b, S.numel())
            if r_eff == 0:
                U_list.append(torch.zeros((mh, 0), dtype=block.dtype, device='cpu'))
                V_list.append(torch.zeros((nh, 0), dtype=block.dtype, device='cpu'))
                continue
            Ur = U[:, :r_eff] * torch.sqrt(S[:r_eff])
            Vr = (Vh[:r_eff, :].T) * torch.sqrt(S[:r_eff])
            U_list.append(Ur.detach().cpu())
            V_list.append(Vr.detach().cpu())

    meta = dict(shape=orig_shape, m=m, n=n, mh=mh, nh=nh, k=k, rank=r_b)
    return U_list, V_list, meta


def _ensure_device(tensor: torch.Tensor, device: Optional[torch.device], dtype: Optional[torch.dtype]):
    """Move tensor to the desired device/dtype if necessary."""

    if device is None and dtype is None:
        return tensor
    target_device = device if device is not None else tensor.device
    target_dtype = dtype if dtype is not None else tensor.dtype
    return tensor.to(device=target_device, dtype=target_dtype)


def bkd_recover(U_list: Iterable[torch.Tensor], V_list: Iterable[torch.Tensor], meta: Dict,
                device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    m, n, mh, nh, k = meta['m'], meta['n'], meta['mh'], meta['nh'], meta['k']
    pad = torch.zeros((mh * k, nh * k), device=device or U_list[0].device, dtype=dtype or U_list[0].dtype)
    it = iter(zip(U_list, V_list))
    for bi in range(k):
        for bj in range(k):
            U, V = next(it)
            U = _ensure_device(U, device, dtype)
            V = _ensure_device(V, device, dtype)
            pad[bi * mh:(bi + 1) * mh, bj * nh:(bj + 1) * nh] = U @ V.T
    return pad[:m, :n].reshape(meta['shape'])


def bkd_aad_recover(U_list: Iterable[torch.Tensor], V_list: Iterable[torch.Tensor], meta: Dict,
                    aad_seed: int, layer_name: str, device: Optional[torch.device] = None,
                    dtype: Optional[torch.dtype] = None):
    m, n, mh, nh, k = meta['m'], meta['n'], meta['mh'], meta['nh'], meta['k']
    rank = meta['rank']
    base_seed = stable_layer_seed(aad_seed, layer_name)
    pad = torch.zeros((mh * k, nh * k), device=device or U_list[0].device, dtype=dtype or U_list[0].dtype)

    it = iter(zip(U_list, V_list))
    for bi in range(k):
        for bj in range(k):
            U, V = next(it)
            U = _ensure_device(U, device, dtype)
            V = _ensure_device(V, device, dtype)
            block_seed = _derive_block_seed(base_seed, layer_name, bi, bj)
            U_tilde, V_tilde = generate_aad_basis(block_seed, mh, nh, rank, device=U.device)
            pad[bi * mh:(bi + 1) * mh, bj * nh:(bj + 1) * nh] = U @ V_tilde.T + U_tilde @ V.T

    return pad[:m, :n].reshape(meta['shape'])


def post_hoc_decomposition(delta_w, rank):
    """
    SVD分解用于压缩
    返回U'=U√S, V'=V√S
    """
    orig_shape = delta_w.shape

    if len(orig_shape) > 2:
        delta_w = delta_w.reshape(orig_shape[0], -1)

    m, n = delta_w.shape
    actual_rank = min(rank, min(m, n))

    # SVD分解
    U, S, Vh = torch.linalg.svd(delta_w, full_matrices=False)

    # 截取前r个分量
    U_r = U[:, :actual_rank]
    S_r = S[:actual_rank]
    Vh_r = Vh[:actual_rank, :]

    # 平方根分配
    S_sqrt = torch.sqrt(S_r + 1e-8)
    U_prime = U_r * S_sqrt.unsqueeze(0)
    V_prime = Vh_r.T * S_sqrt.unsqueeze(0)

    return U_prime, V_prime


def AAD_decomposition(delta_w, rank, seed, lambda_reg=1e-6, n_iter=2):
    """
    AAD分解：ΔW ≈ U*Ṽ^T + Ũ*V^T
    """
    orig_shape = delta_w.shape
    device = delta_w.device

    if len(orig_shape) > 2:
        delta_w = delta_w.reshape(orig_shape[0], -1)

    m, n = delta_w.shape
    actual_rank = min(rank, min(m, n))

    # 生成固定随机基
    U_tilde, V_tilde = generate_aad_basis(seed, m, n, actual_rank, device)

    # 初始化（稳定的solve）
    VtV = V_tilde.T @ V_tilde
    I_v = torch.eye(actual_rank, device=device)
    U = delta_w @ torch.linalg.solve(VtV + lambda_reg * I_v, V_tilde.T).T

    UtU = U_tilde.T @ U_tilde
    I_u = torch.eye(actual_rank, device=device)
    V = delta_w.T @ torch.linalg.solve(UtU + lambda_reg * I_u, U_tilde.T).T

    # ALS迭代
    for _ in range(n_iter):
        target = delta_w - U_tilde @ V.T
        U = target @ torch.linalg.solve(VtV + lambda_reg * I_v, V_tilde.T).T

        target = delta_w.T - V_tilde @ U.T
        V = target @ torch.linalg.solve(UtU + lambda_reg * I_u, U_tilde.T).T

    return U, V


def _layer_dimensions(shape: Iterable[int]):
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return int(shape[0]), 1
    if len(shape) > 2:
        m = int(shape[0])
        n = int(math.prod(shape[1:]))
        return m, n
    return int(shape[0]), int(shape[1])


def allocate_budget(flatten_plan, target_cr: float, mode: str = 'svd', base_floor: int = 1,
                    importance: Optional[Dict[str, float]] = None):
    """
    分配按层压缩预算，自动解 SVD rank 或 BKD blocks。

    Args:
        flatten_plan: build_flatten_plan 生成的列表。
        target_cr: 目标压缩率（>0）。支持 compressed/original (<1) 或 original/compressed (>=1)。
        mode: {'svd', 'aad', 'bkd', 'bkd_aad'}。
        base_floor: 小层最小的 rank / 总秩。
        importance: dict[name] = 重要性权重，可结合 \|ΔW\|_F。

    Returns:
        dict[name] = {'rank': r, 'blocks': k}
    """

    target_cr = float(target_cr)
    if target_cr <= 0:
        return {}

    target_items = [item for item in flatten_plan if item.get('is_target')]
    if not target_items:
        return {}

    records = []
    total_dense = 0
    total_weight = 0.0
    for item in target_items:
        m, n = _layer_dimensions(item['shape'])
        dense = m * n
        if dense == 0:
            continue
        weight = float(dense)
        if importance is not None:
            weight *= max(float(importance.get(item['name'], 0.0)), 0.0) + 1e-12
        records.append({
            'name': item['name'],
            'm': m,
            'n': n,
            'dense': dense,
            'weight': weight,
        })
        total_dense += dense
        total_weight += weight

    if not records or total_dense == 0:
        return {}

    # 支持两种压缩率定义
    if target_cr >= 1.0:
        total_budget = total_dense / target_cr
    else:
        total_budget = total_dense * target_cr

    if total_budget <= 0:
        return {}

    if total_weight <= 0:
        total_weight = float(len(records))
        for rec in records:
            rec['weight'] = 1.0

    per_layer_cfg: Dict[str, Dict[str, int]] = {}

    mode = mode.lower()
    if mode not in {'svd', 'aad', 'bkd', 'bkd_aad'}:
        mode = 'svd'

    if mode in {'svd', 'aad'}:
        denom = {rec['name']: rec['m'] + rec['n'] for rec in records}
        max_rank = {rec['name']: min(rec['m'], rec['n']) for rec in records}
        usage = {}

        for rec in records:
            share = total_budget * (rec['weight'] / total_weight)
            denom_val = denom[rec['name']]
            r = int(share // denom_val) if denom_val > 0 else 0
            r = max(r, base_floor if max_rank[rec['name']] >= base_floor else max_rank[rec['name']])
            r = min(r, max_rank[rec['name']])
            per_layer_cfg[rec['name']] = {'rank': r, 'blocks': 1}
            usage[rec['name']] = denom_val * r

        total_used = sum(usage.values())
        while total_used > total_budget:
            reducible = [name for name, cfg in per_layer_cfg.items()
                         if cfg['rank'] > max(1, base_floor)]
            if not reducible:
                break
            worst = max(reducible, key=lambda n: usage[n])
            if per_layer_cfg[worst]['rank'] <= max(1, base_floor):
                break
            per_layer_cfg[worst]['rank'] -= 1
            usage[worst] -= denom[worst]
            total_used -= denom[worst]

    else:  # BKD / BKD+A
        layer_state = {}
        usage = {}
        candidate_blocks = (1, 2, 3)

        for rec in records:
            share = total_budget * (rec['weight'] / total_weight)
            best = None
            best_cost = None

            for k in candidate_blocks:
                mh = math.ceil(rec['m'] / k)
                nh = math.ceil(rec['n'] / k)
                if mh == 0 or nh == 0:
                    continue
                denom_val = (k * k) * (mh + nh)
                if denom_val == 0:
                    continue
                base_block = max(1, math.ceil(base_floor / k))
                r_b = int(share // denom_val)
                r_b = max(r_b, base_block)
                r_b = min(r_b, min(mh, nh))
                if r_b <= 0:
                    continue
                cost = denom_val * r_b
                diff = abs(cost - share)
                candidate = {
                    'blocks': k,
                    'r_b': r_b,
                    'denom': denom_val,
                    'cost': cost,
                }
                if best is None or diff < best_cost or (diff == best_cost and cost < best['cost']):
                    best = candidate
                    best_cost = diff

            if best is None:
                best = {'blocks': 1, 'r_b': max(1, base_floor), 'denom': rec['m'] + rec['n'], 'cost': share}

            k = best['blocks']
            r_total = best['r_b'] * k
            per_layer_cfg[rec['name']] = {'rank': r_total, 'blocks': k}
            usage[rec['name']] = best['denom'] * best['r_b']
            layer_state[rec['name']] = best

        total_used = sum(usage.values())
        while total_used > total_budget:
            reducible = []
            for name, state in layer_state.items():
                floor_block = max(1, math.ceil(base_floor / state['blocks']))
                if state['r_b'] > floor_block:
                    reducible.append(name)
            if not reducible:
                break
            worst = max(reducible, key=lambda n: usage[n])
            state = layer_state[worst]
            state['r_b'] -= 1
            usage[worst] -= state['denom']
            total_used -= state['denom']
            per_layer_cfg[worst]['rank'] = state['r_b'] * state['blocks']

    return per_layer_cfg
