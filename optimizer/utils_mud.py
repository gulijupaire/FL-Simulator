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