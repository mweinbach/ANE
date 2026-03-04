#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_qwen35_4b_ane import ANEFusedQwenMLP  # noqa: E402


class _DummyKernel:
    def close(self) -> None:
        return

    def run_vec_fp16(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


class _FakeBridge:
    def __init__(self) -> None:
        self.last_hidden_tile: int | None = None
        self.last_spatial: int | None = None
        self._compile_count = 0

    def compile_tiled_fused_swiglu_ffn(self, w1, w3, w2, hidden_tile: int, spatial: int):
        self.last_hidden_tile = hidden_tile
        self.last_spatial = spatial
        self._compile_count += 1
        return _DummyKernel()

    def compile_fused_swiglu_ffn(self, w1, w3, w2, spatial: int):
        self.last_hidden_tile = None
        self.last_spatial = spatial
        self._compile_count += 1
        return _DummyKernel()

    def compile_count(self) -> int:
        return self._compile_count


class _DummyMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):  # pragma: no cover - not used in unit test
        raise NotImplementedError


class AutoShapeWrapperTests(unittest.TestCase):
    def test_auto_shape_changes_tile_for_large_layer(self) -> None:
        bridge = _FakeBridge()
        base = _DummyMLP(dim=2560, hidden=9216)
        wrapper = ANEFusedQwenMLP(
            base=base,
            bridge=bridge,  # type: ignore[arg-type]
            name="model.layers.0.mlp",
            spatial=64,
            mode="mlp_tiled",
            hidden_tile=4096,
            auto_shape=True,
            sram_target_mb=30.0,
            tile_multiple=256,
            min_hidden_tile=512,
        )
        wrapper._compile_if_needed()
        self.assertIsNotNone(bridge.last_hidden_tile)
        self.assertLessEqual(int(bridge.last_hidden_tile), 4096)
        self.assertLessEqual(int(bridge.last_hidden_tile), 2048)
        self.assertIn(int(bridge.last_spatial), {16, 24, 32, 40, 48, 64})

    def test_manual_shape_keeps_requested_values(self) -> None:
        bridge = _FakeBridge()
        base = _DummyMLP(dim=2560, hidden=9216)
        wrapper = ANEFusedQwenMLP(
            base=base,
            bridge=bridge,  # type: ignore[arg-type]
            name="model.layers.0.mlp",
            spatial=48,
            mode="mlp_tiled",
            hidden_tile=2048,
            auto_shape=False,
            sram_target_mb=30.0,
            tile_multiple=256,
            min_hidden_tile=512,
        )
        wrapper._compile_if_needed()
        self.assertEqual(bridge.last_hidden_tile, 2048)
        self.assertEqual(bridge.last_spatial, 48)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
