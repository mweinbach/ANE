#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ane_bridge import ANEBridge


def cpu_swiglu_ffn(x: np.ndarray, w1: np.ndarray, w3: np.ndarray, w2: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        h1 = w1 @ x
        h3 = w3 @ x
        silu = h1 * (1.0 / (1.0 + np.exp(-h1)))
        gate = silu * h3
        y = w2 @ gate
    return y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correctness test for tiled fused ANE SwiGLU-FFN kernel")
    p.add_argument("--dim", type=int, default=2560)
    p.add_argument("--hidden", type=int, default=9216)
    p.add_argument("--tile-hidden", type=int, default=2048)
    p.add_argument("--trials", type=int, default=6)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--spatial", type=int, default=32)
    p.add_argument("--atol", type=float, default=8e-2)
    p.add_argument("--rtol", type=float, default=8e-2)
    p.add_argument(
        "--bridge-lib",
        default=str(Path(__file__).resolve().parents[2] / "bridge" / "libane_bridge.dylib"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    dim = args.dim
    hidden = args.hidden

    w1 = (rng.standard_normal((hidden, dim), dtype=np.float32) * 0.02).astype(np.float32)
    w3 = (rng.standard_normal((hidden, dim), dtype=np.float32) * 0.02).astype(np.float32)
    w2 = (rng.standard_normal((dim, hidden), dtype=np.float32) * 0.02).astype(np.float32)

    bridge = ANEBridge(args.bridge_lib)

    t0 = time.perf_counter()
    kern = bridge.compile_tiled_fused_swiglu_ffn(
        w1,
        w3,
        w2,
        hidden_tile=args.tile_hidden,
        spatial=args.spatial,
    )
    compile_ms = (time.perf_counter() - t0) * 1e3
    print(
        f"compiled tiled fused kernel dim={dim} hidden={hidden} tile_hidden={args.tile_hidden} "
        f"spatial={args.spatial} in {compile_ms:.1f} ms"
    )

    max_abs = 0.0
    max_rel = 0.0
    for i in range(args.trials):
        x = (rng.standard_normal((dim,), dtype=np.float32) * 0.2).astype(np.float32)
        y_ref = cpu_swiglu_ffn(x, w1, w3, w2)

        y_ane = kern.run_vec_fp16(x.astype(np.float16)).astype(np.float32)

        abs_err = np.max(np.abs(y_ane - y_ref))
        denom = np.maximum(np.abs(y_ref), 1e-3)
        rel_err = np.max(np.abs(y_ane - y_ref) / denom)
        max_abs = max(max_abs, float(abs_err))
        max_rel = max(max_rel, float(rel_err))

        print(f"trial {i:02d}: max_abs={abs_err:.5f} max_rel={rel_err:.5f}")

    kern.close()

    ok = (max_abs <= args.atol) or (max_rel <= args.rtol)
    print(
        f"summary: max_abs={max_abs:.5f} max_rel={max_rel:.5f} "
        f"thresholds(atol={args.atol}, rtol={args.rtol}) -> {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
