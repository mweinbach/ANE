#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shape_policy import (  # noqa: E402
    auto_hidden_tile,
    auto_spatial,
    estimate_swiglu_tile_weight_mb,
    estimate_swiglu_weight_mb,
    select_swiglu_shape,
)


class ShapePolicyTests(unittest.TestCase):
    def test_weight_estimates(self) -> None:
        total = estimate_swiglu_weight_mb(2560, 9216)
        self.assertGreater(total, 100.0)
        self.assertLess(total, 150.0)

        tiled = estimate_swiglu_tile_weight_mb(2560, 1536)
        self.assertGreater(tiled, 20.0)
        self.assertLess(tiled, 30.0)

    def test_auto_hidden_tile_stays_within_budget(self) -> None:
        tile = auto_hidden_tile(
            dim=2560,
            hidden=9216,
            sram_target_mb=24.0,
            tile_multiple=256,
            min_hidden_tile=512,
        )
        self.assertEqual(tile % 256, 0)
        self.assertLessEqual(tile, 9216)
        self.assertGreaterEqual(tile, 512)
        self.assertLessEqual(estimate_swiglu_tile_weight_mb(2560, tile), 24.0 + 1e-6)

    def test_auto_hidden_tile_caps_to_hidden(self) -> None:
        tile = auto_hidden_tile(
            dim=768,
            hidden=640,
            sram_target_mb=128.0,
            tile_multiple=256,
            min_hidden_tile=512,
        )
        self.assertLessEqual(tile, 640)
        self.assertGreaterEqual(tile, 1)

    def test_auto_spatial_range(self) -> None:
        for dim, hidden in [(768, 2048), (1024, 4096), (2560, 9216), (4096, 14336)]:
            s = auto_spatial(dim, hidden, preferred_spatial=32)
            self.assertIn(s, {16, 24, 32, 40, 48, 64})

    def test_select_manual_shape(self) -> None:
        choice = select_swiglu_shape(
            dim=2560,
            hidden=9216,
            mode="mlp_tiled",
            requested_spatial=48,
            requested_hidden_tile=2048,
            auto_shape=False,
        )
        self.assertEqual(choice.spatial, 48)
        self.assertEqual(choice.hidden_tile, 2048)
        self.assertFalse(choice.auto_shape)
        self.assertEqual(choice.reason, "manual")

    def test_select_auto_shape_reduces_tile_if_needed(self) -> None:
        manual = select_swiglu_shape(
            dim=2560,
            hidden=9216,
            mode="mlp_tiled",
            requested_spatial=64,
            requested_hidden_tile=4096,
            auto_shape=False,
        )
        auto = select_swiglu_shape(
            dim=2560,
            hidden=9216,
            mode="mlp_tiled",
            requested_spatial=64,
            requested_hidden_tile=4096,
            auto_shape=True,
            sram_target_mb=24.0,
            tile_multiple=256,
            min_hidden_tile=512,
        )
        self.assertLessEqual(auto.hidden_tile, manual.hidden_tile)
        self.assertLessEqual(auto.tile_weight_mb, 24.0 + 1e-6)
        self.assertTrue(auto.auto_shape)
        self.assertTrue(auto.reason.startswith("auto("))


if __name__ == "__main__":
    raise SystemExit(unittest.main())
