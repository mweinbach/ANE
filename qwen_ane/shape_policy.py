from __future__ import annotations

from dataclasses import dataclass
import math

_MB = 1024.0 * 1024.0


def estimate_swiglu_weight_mb(dim: int, hidden: int) -> float:
    if dim < 1 or hidden < 1:
        raise ValueError("dim and hidden must be >= 1")
    # 3 matrices in fused SwiGLU FFN path, fp16 storage.
    return (3.0 * dim * hidden * 2.0) / _MB


def estimate_swiglu_tile_weight_mb(dim: int, tile_hidden: int) -> float:
    if dim < 1 or tile_hidden < 1:
        raise ValueError("dim and tile_hidden must be >= 1")
    return (3.0 * dim * tile_hidden * 2.0) / _MB


def _align_down(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return (value // multiple) * multiple


def auto_hidden_tile(
    dim: int,
    hidden: int,
    sram_target_mb: float = 30.0,
    tile_multiple: int = 256,
    min_hidden_tile: int = 512,
) -> int:
    if dim < 1 or hidden < 1:
        raise ValueError("dim and hidden must be >= 1")
    if sram_target_mb <= 0:
        raise ValueError("sram_target_mb must be > 0")
    if tile_multiple < 1:
        raise ValueError("tile_multiple must be >= 1")
    if min_hidden_tile < 1:
        raise ValueError("min_hidden_tile must be >= 1")

    bytes_budget = int(sram_target_mb * _MB)
    max_tile_by_budget = max(1, bytes_budget // int(3 * dim * 2))

    tile = min(hidden, max_tile_by_budget)
    tile = _align_down(tile, tile_multiple)
    if tile < 1:
        tile = min(hidden, tile_multiple)
    if tile < 1:
        tile = 1

    if hidden >= min_hidden_tile:
        min_aligned = _align_down(min_hidden_tile, tile_multiple)
        if min_aligned < 1:
            min_aligned = min_hidden_tile
        tile = max(tile, min_aligned)

    return max(1, min(hidden, tile))


def auto_spatial(dim: int, hidden: int, preferred_spatial: int = 32) -> int:
    if dim < 1 or hidden < 1:
        raise ValueError("dim and hidden must be >= 1")
    if preferred_spatial < 1:
        raise ValueError("preferred_spatial must be >= 1")

    allowed = [16, 24, 32, 40, 48, 64]

    if dim >= 4096:
        target = 24
    elif dim >= 2048:
        target = 32
    elif dim >= 1024:
        target = 40
    else:
        target = 64

    # Keep preference as a soft ceiling for large dims; small dims can still
    # benefit from larger spatial packing to reduce dispatch overhead.
    if dim >= 2048:
        target = min(target, max(16, preferred_spatial))
    else:
        target = max(target, min(64, preferred_spatial))

    return min(allowed, key=lambda v: (abs(v - target), v))


@dataclass(frozen=True)
class ANEShapeChoice:
    spatial: int
    hidden_tile: int
    tile_weight_mb: float
    total_weight_mb: float
    tile_count: int
    auto_shape: bool
    reason: str


def select_swiglu_shape(
    dim: int,
    hidden: int,
    mode: str,
    requested_spatial: int,
    requested_hidden_tile: int,
    auto_shape: bool,
    sram_target_mb: float = 30.0,
    tile_multiple: int = 256,
    min_hidden_tile: int = 512,
) -> ANEShapeChoice:
    if dim < 1 or hidden < 1:
        raise ValueError("dim and hidden must be >= 1")
    if requested_spatial < 1:
        raise ValueError("requested_spatial must be >= 1")
    if requested_hidden_tile < 1:
        raise ValueError("requested_hidden_tile must be >= 1")

    total_weight_mb = estimate_swiglu_weight_mb(dim, hidden)

    if not auto_shape:
        tile = min(requested_hidden_tile, hidden)
        tile_weight_mb = estimate_swiglu_tile_weight_mb(dim, tile)
        tile_count = int(math.ceil(hidden / tile))
        return ANEShapeChoice(
            spatial=requested_spatial,
            hidden_tile=tile,
            tile_weight_mb=tile_weight_mb,
            total_weight_mb=total_weight_mb,
            tile_count=tile_count,
            auto_shape=False,
            reason="manual",
        )

    spatial = auto_spatial(dim, hidden, preferred_spatial=requested_spatial)
    if mode == "mlp_tiled":
        budget_tile = auto_hidden_tile(
            dim=dim,
            hidden=hidden,
            sram_target_mb=sram_target_mb,
            tile_multiple=tile_multiple,
            min_hidden_tile=min_hidden_tile,
        )
        tile = min(requested_hidden_tile, budget_tile)
    else:
        tile = min(requested_hidden_tile, hidden)

    tile = max(1, min(tile, hidden))
    tile_weight_mb = estimate_swiglu_tile_weight_mb(dim, tile)
    tile_count = int(math.ceil(hidden / tile))
    reason = (
        f"auto(sram_target_mb={sram_target_mb:.1f}, tile_multiple={tile_multiple}, "
        f"min_hidden_tile={min_hidden_tile})"
    )
    return ANEShapeChoice(
        spatial=spatial,
        hidden_tile=tile,
        tile_weight_mb=tile_weight_mb,
        total_weight_mb=total_weight_mb,
        tile_count=tile_count,
        auto_shape=True,
        reason=reason,
    )
