"""Utility helpers to wrap standard modules with DMU layers."""

from dataclasses import dataclass
from typing import List, Tuple

import torch.nn as nn

from optimizer.utils_mud import stable_layer_seed
from .dmu import DMU_Conv2d, DMU_Linear


@dataclass
class DMUConfig:
    """Configuration for DMU wrapping."""

    rank: int
    init_mag: float
    pattern: str = "ab"
    skip_first: int = 0
    skip_last: int = 0

    def validate(self) -> None:
        if self.rank <= 0:
            raise ValueError("DMU rank must be positive")
        if self.init_mag < 0:
            raise ValueError("DMU init_mag must be non-negative")
        if self.skip_first < 0 or self.skip_last < 0:
            raise ValueError("skip counts must be non-negative")


def _collect_target_layers(model: nn.Module) -> List[Tuple[nn.Module, str, nn.Module, str]]:
    """Collect all Linear/Conv2d layers with their parents and qualified names."""

    targets: List[Tuple[nn.Module, str, nn.Module, str]] = []

    def recurse(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                targets.append((module, name, child, child_prefix))
            else:
                recurse(child, child_prefix)

    recurse(model)
    return targets


def dmu_wrap(model: nn.Module, cfg: DMUConfig, seed: int) -> nn.Module:
    """Replace Linear/Conv2d layers with their DMU counterparts."""

    cfg.validate()

    targets = _collect_target_layers(model)
    if not targets:
        return model

    total = len(targets)
    start = min(cfg.skip_first, total)
    end = total - min(cfg.skip_last, total - start)

    for index, (parent, name, layer, qualified_name) in enumerate(targets):
        if index < start or index >= end:
            continue

        layer_seed = stable_layer_seed(seed, qualified_name)
        if isinstance(layer, nn.Linear):
            base_weight = layer.weight.detach().clone()
            base_bias = None if layer.bias is None else layer.bias.detach().clone()
            bias_requires_grad = False if layer.bias is None else layer.bias.requires_grad
            wrapped = DMU_Linear(
                layer.in_features,
                layer.out_features,
                base=base_weight,
                bias=base_bias,
                bias_requires_grad=bias_requires_grad,
                rank=cfg.rank,
                init_mag=cfg.init_mag,
                pattern=cfg.pattern,
                seed=layer_seed,
            )
        else:  # nn.Conv2d
            wrapped = DMU_Conv2d(
                layer,
                rank=cfg.rank,
                init_mag=cfg.init_mag,
                pattern=cfg.pattern,
                seed=layer_seed,
            )
        setattr(parent, name, wrapped)

    return model
