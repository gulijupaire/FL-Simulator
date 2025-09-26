"""Deterministic Matrix Update (DMU) modules for parametric MUD."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class MatUpdate(nn.Module):
    """Low-rank matrix update with optional anchored components."""

    _ALLOWED_PATTERNS = {"ab", "fab", "fab+cfd"}

    def __init__(self, in_f: int, out_f: int, rank: int, init_mag: float,
                 pattern: str = "ab", seed: int = 0):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for MatUpdate")
        if pattern not in self._ALLOWED_PATTERNS:
            raise ValueError(f"Unsupported pattern: {pattern}")

        self.in_f, self.out_f, self.r = int(in_f), int(out_f), int(rank)
        self.pattern = pattern

        # Trainable factors
        self.U = nn.Parameter(torch.zeros(self.out_f, self.r))
        self.V = nn.Parameter(torch.zeros(self.in_f, self.r))

        # Fixed anchors (for AAD / BKD variants)
        g = torch.Generator().manual_seed(int(seed))
        self.Ut = nn.Parameter(torch.randn(self.out_f, self.r, generator=g), requires_grad=False)
        self.Vt = nn.Parameter(torch.randn(self.in_f, self.r, generator=g), requires_grad=False)

        self.reset(init_mag)

    def reset(self, init_mag: float) -> None:
        """Re-initialise the trainable factors."""

        nn.init.uniform_(self.U, -float(init_mag), float(init_mag))
        nn.init.zeros_(self.V)

    def forward_update(self) -> torch.Tensor:
        """Return the current low-rank update matrix."""

        if self.pattern == "ab":  # standard low-rank U V^T
            return self.U @ self.V.T
        if self.pattern == "fab":  # train right, fixed left => Ut V^T
            return self.Ut @ self.V.T
        if self.pattern == "fab+cfd":  # Ut V^T + U Vt^T
            return self.Ut @ self.V.T + self.U @ self.Vt.T
        raise ValueError(f"Unsupported pattern: {self.pattern}")


class DMU_Linear(nn.Module):
    """Linear layer augmented with a deterministic matrix update."""

    def __init__(self, in_f: int, out_f: int, *, base: Optional[torch.Tensor] = None,
                 bias: Optional[torch.Tensor] = None, bias_requires_grad: bool = True,
                 rank: int, init_mag: float, pattern: str = "ab", seed: int = 0):
        super().__init__()

        self.in_features = in_f
        self.out_features = out_f

        weight = torch.zeros(out_f, in_f) if base is None else base.detach().clone()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=bias_requires_grad)

        self.update = MatUpdate(in_f, out_f, rank=rank, init_mag=init_mag, pattern=pattern, seed=seed)

    @torch.no_grad()
    def push_reset_update(self, seed: int, init_mag: float) -> None:
        """Accumulate the current update into the base weight and reset factors."""

        self.weight.add_(self.update.forward_update())
        rank, pattern = self.update.r, self.update.pattern
        device, dtype = self.weight.device, self.weight.dtype
        self.update = MatUpdate(self.weight.shape[1], self.weight.shape[0],
                                rank=rank, init_mag=init_mag, pattern=pattern, seed=seed)
        self.update.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight + self.update.forward_update(), self.bias)


class DMU_Conv2d(nn.Module):
    """Conv2d layer augmented with a deterministic matrix update."""

    def __init__(self, conv: nn.Conv2d, *, rank: int, init_mag: float,
                 pattern: str = "ab", seed: int = 0):
        super().__init__()

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = getattr(conv, "padding_mode", "zeros")
        self._reversed_padding_repeated_twice = getattr(
            conv, "_reversed_padding_repeated_twice", [0, 0, 0, 0]
        )

        weight = conv.weight.detach().clone()
        self.weight = nn.Parameter(weight, requires_grad=False)

        if conv.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(conv.bias.detach().clone(), requires_grad=conv.bias.requires_grad)

        in_features = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        self.update = MatUpdate(in_features, self.out_channels, rank=rank,
                                init_mag=init_mag, pattern=pattern, seed=seed)

    @torch.no_grad()
    def push_reset_update(self, seed: int, init_mag: float) -> None:
        update = self.update.forward_update().view_as(self.weight)
        self.weight.add_(update)

        rank, pattern = self.update.r, self.update.pattern
        device, dtype = self.weight.device, self.weight.dtype
        in_features = self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3]
        self.update = MatUpdate(in_features, self.out_channels, rank=rank,
                                init_mag=init_mag, pattern=pattern, seed=seed)
        self.update.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.update.forward_update().view_as(self.weight)
        weight = self.weight + update

        if self.padding_mode != "zeros":
            padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(padded, weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
