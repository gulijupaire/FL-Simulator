import torch



def param_to_vector(model):
    # model parameters ---> vector (same storage)
    vec = []
    for param in model.parameters():
        vec.append(param.reshape(-1))
    return torch.cat(vec)



def get_params_list_with_shape(model, param_list, device):
    vec_with_shape = []
    idx = 0
    for param in model.parameters():
        length = param.numel()
        vec_with_shape.append(param_list[idx:idx + length].reshape(param.shape).to(device))
        idx += length
    return vec_with_shape


def get_mdl_params(model, device=None):
    """
    将模型参数展平成单个向量（不跟踪梯度），可选迁移到 device
    """
    vec = torch.nn.utils.parameters_to_vector([p.detach() for p in model.parameters()])
    return vec.to(device) if device is not None else vec

def set_mdl_params(model, vec):
    """
    用向量形式参数覆盖到模型（in-place）
    """
    device = next(model.parameters()).device
    vec = vec.detach().to(device)
    torch.nn.utils.vector_to_parameters(vec, model.parameters())


