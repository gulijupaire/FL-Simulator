import torch
import math
from functools import reduce
from operator import mul


def tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def count_tensor_bytes(obj) -> int:
    """递归统计任意嵌套(list/tuple/dict)中的 torch.Tensor 字节数。"""
    if torch.is_tensor(obj):
        return tensor_nbytes(obj)
    if isinstance(obj, dict):
        return sum(count_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(count_tensor_bytes(v) for v in obj)
    return 0


def shape_numel(shape) -> int:
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    if isinstance(shape, (list, tuple)):
        return int(reduce(mul, shape, 1))
    return int(shape)  # 已是 numel


def dense_bytes_from_shapes(shapes, dtype=torch.float32) -> int:
    """把一组参数shape当作稠密张量传输，估算总字节数。"""
    elem_size = torch.tensor([], dtype=dtype).element_size()
    return int(elem_size * sum(shape_numel(s) for s in shapes))
