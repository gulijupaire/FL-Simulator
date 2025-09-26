"""Deterministic Matrix Update (DMU) modules for parametric MUD."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .dmu_init import GLOBAL_INIT_STATS, init_pair_, make_generator_from_seed


class MatUpdate(nn.Module):
    """Low-rank matrix update with optional anchored components."""

    _ALLOWED_PATTERNS = {"ab", "fab", "fab+cfd"}

    def __init__(
        self,
        in_f: int,
        out_f: int,
        rank: int,
        init_mag: float,
        pattern: str = "ab",
        seed: int = 0,
        *,
        init_dist: str = "uniform",
        rank_scale: str = "r_quarter",
        stats=GLOBAL_INIT_STATS,
        layer_name: Optional[str] = None,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive for MatUpdate")
        if pattern not in self._ALLOWED_PATTERNS:
            raise ValueError(f"Unsupported pattern: {pattern}")

        self.in_f, self.out_f = int(in_f), int(out_f)
        self.r = min(int(rank), self.in_f, self.out_f)
        self.pattern = pattern

        self.init_dist = init_dist
        self.rank_scale_mode = rank_scale
        self.init_mag = float(init_mag)
        self._stats = stats
        self._layer_name = layer_name

        # Trainable factors
        self.U = nn.Parameter(torch.zeros(self.out_f, self.r))
        self.V = nn.Parameter(torch.zeros(self.in_f, self.r))

        # Fixed anchors (for AAD / BKD variants)
        anchor_gen = make_generator_from_seed(seed)
        self.Ut = nn.Parameter(
            torch.randn(self.out_f, self.r, generator=anchor_gen), requires_grad=False
        )
        self.Vt = nn.Parameter(
            torch.randn(self.in_f, self.r, generator=anchor_gen), requires_grad=False
        )

        self.reset(init_mag=init_mag, generator=make_generator_from_seed(seed), event="init")

    def reset(
        self,
        init_mag: Optional[float] = None,
        *,
        generator: Optional[torch.Generator] = None,
        event: str = "init",
    ) -> None:
        """Re-initialise the trainable factors."""

        if init_mag is not None:
            self.init_mag = float(init_mag)
        init_pair_(
            self.U,
            self.V,
            self.init_dist,
            self.init_mag,
            self.r,
            self.rank_scale_mode,
            generator=generator,
            stats=self._stats,
            layer=self._layer_name,
            event=event,
        )

    def forward_update(self) -> torch.Tensor:
        """Return the current low-rank update matrix respecting the chosen pattern."""

        if self.pattern == "ab":  # standard low-rank U V^T
            return self.U @ self.V.T
        if self.pattern == "fab":  # train right, fixed left => Ut V^T
            return self.Ut @ self.V.T
        if self.pattern == "fab+cfd":  # Ut V^T + U Vt^T
            return self.Ut @ self.V.T + self.U @ self.Vt.T
        raise ValueError(f"Unsupported pattern: {self.pattern}")


@torch.no_grad()
def _reset_update_inplace(update: "MatUpdate", seed: int, init_mag: float) -> None:
    """Reset a MatUpdate instance in-place without replacing Parameter objects."""

    device = update.Ut.device
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    trainable_gen = make_generator_from_seed(seed, device=update.U.device)
    if init_mag is not None:
        update.init_mag = float(init_mag)

    if update.Ut.device == update.U.device:
        anchor_gen = trainable_gen
    else:
        anchor_gen = make_generator_from_seed(seed, device=update.Ut.device)

    update.Ut.copy_(torch.randn_like(update.Ut, generator=anchor_gen))
    update.Vt.copy_(torch.randn_like(update.Vt, generator=anchor_gen))

    update.reset(generator=trainable_gen, event="reset")


class DMU_Linear(nn.Module):
    """Linear layer augmented with a deterministic matrix update."""

    def __init__(
        self,
        in_f: int,
        out_f: int,
        *,
        base: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        bias_requires_grad: bool = True,
        rank: int,
        init_mag: float,
        pattern: str = "ab",
        seed: int = 0,
        init_dist: str = "uniform",
        rank_scale: str = "r_quarter",
        stats=GLOBAL_INIT_STATS,
        layer_name: Optional[str] = None,
    ):
        super().__init__()

        self.in_features = in_f
        self.out_features = out_f

        weight = torch.zeros(out_f, in_f) if base is None else base.detach().clone()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=bias_requires_grad)

        self.update = MatUpdate(
            in_f,
            out_f,
            rank=rank,
            init_mag=init_mag,
            pattern=pattern,
            seed=seed,
            init_dist=init_dist,
            rank_scale=rank_scale,
            stats=stats,
            layer_name=layer_name,
        )

    @torch.no_grad()
    def push_reset_update(self, seed: int, init_mag: Optional[float] = None) -> None:
        """Accumulate the current update into the base weight and reset factors."""

        self.weight.add_(self.update.forward_update())
        _reset_update_inplace(self.update, seed, init_mag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight + self.update.forward_update(), self.bias)


class DMU_Conv2d(nn.Module):
    """Conv2d layer augmented with a deterministic matrix update."""

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        rank: int,
        init_mag: float,
        pattern: str = "ab",
        seed: int = 0,
        init_dist: str = "uniform",
        rank_scale: str = "r_quarter",
        stats=GLOBAL_INIT_STATS,
        layer_name: Optional[str] = None,
    ):
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
        self.update = MatUpdate(
            in_features,
            self.out_channels,
            rank=rank,
            init_mag=init_mag,
            pattern=pattern,
            seed=seed,
            init_dist=init_dist,
            rank_scale=rank_scale,
            stats=stats,
            layer_name=layer_name,
        )

    @torch.no_grad()
    def push_reset_update(self, seed: int, init_mag: Optional[float] = None) -> None:
        update = self.update.forward_update().view_as(self.weight)
        self.weight.add_(update)
        _reset_update_inplace(self.update, seed, init_mag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.update.forward_update().view_as(self.weight)
        weight = self.weight + update

        if self.padding_mode != "zeros":
            padded = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            return F.conv2d(padded, weight, self.bias, self.stride, _pair(0), self.dilation, self.groups)

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
