#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_qwen35_4b_ane import _fallback_model_candidates_for, _looks_like_mlx_quantized_model  # noqa: E402


class ModelFallbackTests(unittest.TestCase):
    def test_detects_mlx_community_model_ids(self) -> None:
        self.assertTrue(_looks_like_mlx_quantized_model("mlx-community/Qwen3.5-4B-mxfp4"))
        self.assertTrue(
            _looks_like_mlx_quantized_model(
                "/Users/me/.cache/huggingface/hub/models--mlx-community--Qwen3.5-2B-6bit/snapshots/abc"
            )
        )

    def test_non_mlx_model_has_no_fallback_candidates(self) -> None:
        self.assertFalse(_looks_like_mlx_quantized_model("Qwen/Qwen3.5-4B"))
        self.assertEqual(_fallback_model_candidates_for("Qwen/Qwen3.5-4B"), [])

    def test_mlx_model_includes_qwen4b_repo_fallback(self) -> None:
        candidates = _fallback_model_candidates_for("mlx-community/Qwen3.5-2B-6bit")
        self.assertIn("Qwen/Qwen3.5-4B", candidates)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
