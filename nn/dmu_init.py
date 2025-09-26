import re
from typing import Dict, List, Optional, Tuple

import torch


def parse_init_spec(spec: str) -> Tuple[str, float]:
    """Parse an initialisation specification string.

    Accepted forms include "uni_0.3", "uniform:0.3", "nor_0.1", "normal:0.1".
    Returns a tuple ``(distribution, magnitude)`` with distribution either
    ``"uniform"`` or ``"normal"``. Fallback is a uniform distribution with
    magnitude 0.3.
    """
    if not isinstance(spec, str):
        return "uniform", float(spec)
    s = spec.strip().lower().replace("-", "_").replace(":", "_")
    m = re.match(r"(uni|uniform)_(\d*\.?\d+)", s)
    if m:
        return "uniform", float(m.group(2))
    m = re.match(r"(nor|normal)_(\d*\.?\d+)", s)
    if m:
        return "normal", float(m.group(2))
    try:
        return "uniform", float(s)
    except Exception:
        return "uniform", 0.3


def rank_scale(rank: int, mode: str) -> float:
    """Return the scaling factor for rank-dependent initialisation."""
    if rank is None or rank <= 1:
        return 1.0
    m = (mode or "r_quarter").lower()
    if m in ("r_quarter", "r-0.25"):
        return rank ** (-0.25)
    if m in ("r_sqrt", "r-0.5"):
        return rank ** (-0.5)
    return 1.0


def init_tensor_(
    t: torch.Tensor,
    dist: str,
    mag: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> None:
    if dist == "normal":
        torch.nn.init.normal_(t, mean=0.0, std=mag, generator=generator)
    else:
        torch.nn.init.uniform_(t, a=-mag, b=mag, generator=generator)


def init_pair_(
    U: torch.Tensor,
    V: torch.Tensor,
    dist: str,
    mag: float,
    rank: int,
    scale_mode: str,
    *,
    generator: Optional[torch.Generator] = None,
    stats: Optional["InitStats"] = None,
    layer: Optional[str] = None,
    event: str = "init",
) -> None:
    """Initialise a pair of tensors representing low-rank factors."""
    scaled = mag * rank_scale(rank, scale_mode)
    init_tensor_(U, dist, scaled, generator=generator)
    init_tensor_(V, dist, scaled, generator=generator)
    if stats is not None:
        stats.record(event, layer, dist, scaled, rank, scale_mode, U)
        stats.record(event, layer, dist, scaled, rank, scale_mode, V)


def make_generator_from_seed(
    seed: int,
    device: Optional[torch.device] = None,
) -> torch.Generator:
    kwargs: Dict[str, object] = {}
    if device is not None:
        kwargs["device"] = device
    g = torch.Generator(**kwargs)
    g.manual_seed(int(seed) & 0x7FFFFFFF)
    return g


class InitStats:
    """Light-weight statistics collector for DMU initialisation events."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self._records: Dict[str, List[Dict[str, float]]] = {"init": [], "reset": []}

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.clear()

    def clear(self) -> None:
        for key in self._records:
            self._records[key].clear()

    def record(
        self,
        event: str,
        layer: Optional[str],
        dist: str,
        mag: float,
        rank: int,
        scale_mode: str,
        tensor: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return
        event = event if event in self._records else "init"
        if tensor.numel() == 0:
            return
        with torch.no_grad():
            flat = tensor.detach().reshape(-1).to(torch.float64)
            mean_val = flat.mean().item()
            std_val = flat.std(unbiased=False).item()
        self._records[event].append(
            {
                "dist": 0 if dist == "uniform" else 1,
                "mag": float(mag),
                "rank": float(rank),
                "scale": 0 if scale_mode == "none" else (1 if scale_mode == "r_quarter" else 2),
                "mean": float(mean_val),
                "std": float(std_val),
            }
        )

    def summary_lines(self) -> List[str]:
        if not self.enabled:
            return []
        lines: List[str] = []
        mapping = {0: "uniform", 1: "normal"}
        scale_map = {0: "none", 1: "r_quarter", 2: "r_sqrt"}
        for event, recs in self._records.items():
            if not recs:
                lines.append(f"[DMU stats] {event}: no records.")
                continue
            count = len(recs)
            avg_std = sum(r["std"] for r in recs) / count
            avg_abs_mean = sum(abs(r["mean"]) for r in recs) / count
            dists = {mapping.get(int(r["dist"]), "uniform") for r in recs}
            scales = {scale_map.get(int(r["scale"]), "r_quarter") for r in recs}
            lines.append(
                "[DMU stats] {event}: {count} tensors | dist={dist} | scale={scale} | "
                "mean(|mean|)={m:.4e} | mean(std)={s:.4e}".format(
                    event=event,
                    dist=",".join(sorted(dists)),
                    scale=",".join(sorted(scales)),
                    m=avg_abs_mean,
                    s=avg_std,
                )
            )
        return lines


GLOBAL_INIT_STATS = InitStats()
