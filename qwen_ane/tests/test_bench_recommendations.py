#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_ane_decode_kernels import BenchResult, best_results_by_shape  # noqa: E402


class BenchRecommendationTests(unittest.TestCase):
    def test_best_results_by_shape(self) -> None:
        rows = [
            BenchResult(2560, 9216, 32, "fused_tiled", 1536, 100.0, 2.0, 70.0, 0.07, 0.44),
            BenchResult(2560, 9216, 48, "fused_tiled", 1536, 100.0, 1.5, 90.0, 0.09, 0.57),
            BenchResult(1024, 4096, 64, "fused_mlp", 4096, 50.0, 1.2, 60.0, 0.06, 0.38),
            BenchResult(1024, 4096, 32, "fused_mlp", 4096, 50.0, 1.0, 72.0, 0.072, 0.46),
        ]
        best = best_results_by_shape(rows)
        self.assertEqual(set(best.keys()), {(2560, 9216), (1024, 4096)})
        self.assertEqual(best[(2560, 9216)].spatial, 48)
        self.assertEqual(best[(1024, 4096)].spatial, 32)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
