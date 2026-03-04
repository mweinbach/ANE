#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_backend import _persist_autotune_result  # noqa: E402


class AutoTuneCacheTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            model_id="mlx-community/Qwen3.5-4B-mxfp4",
            prefill_device="mps",
            ane_mode="mlp_tiled",
            ane_spatial=32,
            ane_hidden_tile=2048,
            ane_shape_policy="auto",
            dtype="fp16",
            kv_cache_dtype="auto",
        )

    def _result(self, eval_ms: float) -> dict:
        return {
            "shape": {"module": "m.layers.0.mlp", "dim": 2560, "hidden": 9216, "weight_source": "synthetic"},
            "mode": "mlp_tiled",
            "best": {"spatial": 32, "tile_hidden": 2048, "eval_ms": eval_ms},
            "recommended": {"ane_spatial": 32, "ane_hidden_tile": 2048},
            "candidates": [],
            "failures": [],
            "bridge_compiles": 1,
        }

    def test_persist_autotune_result_writes_cache_and_tracks_best(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autotune-cache-test-") as tmpdir:
            cache_path = Path(tmpdir) / "autotune_runs.json"
            runtime_meta = {
                "source_model_id": "mlx-community/Qwen3.5-4B-mxfp4",
                "runtime_model_id": "Qwen/Qwen3.5-4B",
            }
            with mock.patch("gui_backend._autotune_cache_path", return_value=cache_path):
                _path, key, is_best_first = _persist_autotune_result(
                    result=self._result(eval_ms=2.0),
                    args=self._args(),
                    runtime_meta=runtime_meta,
                    req_id="a",
                )
                self.assertTrue(is_best_first)
                _path, same_key, is_best_second = _persist_autotune_result(
                    result=self._result(eval_ms=3.0),
                    args=self._args(),
                    runtime_meta=runtime_meta,
                    req_id="b",
                )
                self.assertEqual(key, same_key)
                self.assertFalse(is_best_second)

            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            self.assertIn("entries", payload)
            self.assertEqual(len(payload["entries"]), 2)
            self.assertIn("best_by_key", payload)
            self.assertIn(key, payload["best_by_key"])
            self.assertEqual(payload["best_by_key"][key]["best_eval_ms"], 2.0)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
