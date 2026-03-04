#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gui_backend import _find_first_qwen_mlp  # noqa: E402


class _ModelWithMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.mlp.up_proj = nn.Linear(dim, hidden, bias=False)
        self.mlp.down_proj = nn.Linear(hidden, dim, bias=False)


class _ModelWithPackedMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.mlp.up_proj = nn.Linear(dim, hidden, bias=False)
        self.mlp.down_proj = nn.Linear(hidden, dim, bias=False)
        # Simulate packed mxfp4-like shape mismatch: in/out metadata stays full,
        # but weight tensors are compressed on the second axis.
        self.mlp.gate_proj.weight = nn.Parameter(torch.zeros((hidden, max(1, dim // 8)), dtype=torch.float32))
        self.mlp.up_proj.weight = nn.Parameter(torch.zeros((hidden, max(1, dim // 8)), dtype=torch.float32))
        self.mlp.down_proj.weight = nn.Parameter(torch.zeros((dim, max(1, hidden // 8)), dtype=torch.float32))


class _ModelWithTransposedDown(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.mlp.up_proj = nn.Linear(dim, hidden, bias=False)
        # Intentionally transposed vs expected [dim, hidden].
        self.mlp.down_proj = nn.Linear(dim, hidden, bias=False)


class GuiBackendAutotuneWeightTests(unittest.TestCase):
    def test_prefers_model_weights_when_shapes_match(self) -> None:
        model = _ModelWithMLP(dim=32, hidden=64)
        _name, w1, w3, w2, dim, hidden, source = _find_first_qwen_mlp(model, seed=7)
        self.assertEqual((dim, hidden), (32, 64))
        self.assertEqual(w1.shape, (64, 32))
        self.assertEqual(w3.shape, (64, 32))
        self.assertEqual(w2.shape, (32, 64))
        self.assertEqual(source, "model_weights")

    def test_uses_synthetic_weights_for_packed_shapes(self) -> None:
        model = _ModelWithPackedMLP(dim=32, hidden=64)
        _name, w1, w3, w2, dim, hidden, source = _find_first_qwen_mlp(model, seed=7)
        self.assertEqual((dim, hidden), (32, 64))
        self.assertEqual(w1.shape, (64, 32))
        self.assertEqual(w3.shape, (64, 32))
        self.assertEqual(w2.shape, (32, 64))
        self.assertTrue(source.startswith("synthetic_from_features"))

    def test_transposes_down_when_only_down_is_transposed(self) -> None:
        model = _ModelWithTransposedDown(dim=16, hidden=24)
        _name, w1, _w3, w2, dim, hidden, source = _find_first_qwen_mlp(model, seed=9)
        self.assertEqual((dim, hidden), (16, 24))
        self.assertEqual(w1.shape, (24, 16))
        self.assertEqual(w2.shape, (16, 24))
        self.assertIn("transposed down", source)


if __name__ == "__main__":
    raise SystemExit(unittest.main())
