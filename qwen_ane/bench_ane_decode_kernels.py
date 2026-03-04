#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ane_bridge import ANEBridge, ANEKernel
from shape_policy import select_swiglu_shape


@dataclass
class BenchResult:
    dim: int
    hidden: int
    spatial: int
    mode: str
    tile_hidden: int
    compile_ms: float
    eval_ms: float
    gflops: float
    tflops: float
    util_pct: float
    note: str = ""


def parse_shape(s: str) -> Tuple[int, int]:
    if ":" not in s:
        raise ValueError(f"shape must be dim:hidden, got {s}")
    a, b = s.split(":", 1)
    return int(a), int(b)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark ANE decode kernels for utilization mapping")
    p.add_argument(
        "--shapes",
        nargs="+",
        default=["768:2048", "1024:4096", "2560:9216"],
        help="List of dim:hidden shapes",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--spatial", type=int, default=32, help="Single spatial value if --spatial-list is omitted")
    p.add_argument(
        "--spatial-list",
        nargs="+",
        type=int,
        default=None,
        help="Optional spatial sweep. Example: --spatial-list 16 32 64",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        choices=["linear_stack", "fused_mlp", "fused_tiled"],
        default=["linear_stack", "fused_mlp", "fused_tiled"],
        help="Decode kernel modes to benchmark",
    )
    p.add_argument("--tile-hidden", type=int, default=2048, help="Hidden tile size for fused_tiled mode")
    p.add_argument(
        "--tile-hidden-list",
        nargs="+",
        type=int,
        default=None,
        help="Optional hidden tile sweep for fused_tiled mode. Example: --tile-hidden-list 1024 1536 2048",
    )
    p.add_argument(
        "--ane-shape-policy",
        choices=["auto", "manual"],
        default="auto",
        help="Shape recommendation policy used for suggested configs",
    )
    p.add_argument("--ane-sram-target-mb", type=float, default=30.0)
    p.add_argument("--ane-tile-multiple", type=int, default=256)
    p.add_argument("--ane-min-hidden-tile", type=int, default=512)
    p.add_argument("--json-out", default="", help="Optional JSONL output path for result rows")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--peak-tflops", type=float, default=15.8)
    p.add_argument(
        "--bridge-lib",
        default=str(Path(__file__).resolve().parents[1] / "bridge" / "libane_bridge.dylib"),
    )
    return p.parse_args()


def bench_kernel(k: ANEKernel, x: np.ndarray, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = k.run_vec_fp16(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = k.run_vec_fp16(x)
    return (time.perf_counter() - t0) * 1e3 / iters


def flops_metrics(dim: int, hidden: int, eval_ms: float, peak_tflops: float) -> tuple[float, float, float]:
    flops = 6.0 * dim * hidden  # 3 matmuls at seq=1
    gflops = flops / (eval_ms * 1e6)
    tflops = gflops / 1000.0
    util_pct = 100.0 * tflops / peak_tflops
    return gflops, tflops, util_pct


def report(res: BenchResult) -> str:
    suffix = f" {res.note}" if res.note else ""
    return (
        f"dim={res.dim:5d} hidden={res.hidden:5d} spatial={res.spatial:3d} mode={res.mode:12s} "
        f"tile={res.tile_hidden:5d} "
        f"compile={res.compile_ms:8.1f}ms eval={res.eval_ms:8.3f}ms "
        f"gflops={res.gflops:8.2f} tflops={res.tflops:6.3f} util={res.util_pct:6.2f}%{suffix}"
    )


def bench_linear_stack(
    bridge: ANEBridge,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
    x: np.ndarray,
    spatial: int,
    warmup: int,
    iters: int,
    peak_tflops: float,
) -> BenchResult:
    t0 = time.perf_counter()
    k1 = bridge.compile_linear(w1, spatial=spatial)
    k3 = bridge.compile_linear(w3, spatial=spatial)
    k2 = bridge.compile_linear(w2, spatial=spatial)
    compile_ms = (time.perf_counter() - t0) * 1e3
    try:
        for _ in range(warmup):
            h1 = k1.run_vec_fp16(x)
            h3 = k3.run_vec_fp16(x)
            gate = (
                h1.astype(np.float32)
                * (1.0 / (1.0 + np.exp(-h1.astype(np.float32))))
                * h3.astype(np.float32)
            ).astype(np.float16)
            _ = k2.run_vec_fp16(gate)

        t0 = time.perf_counter()
        for _ in range(iters):
            h1 = k1.run_vec_fp16(x)
            h3 = k3.run_vec_fp16(x)
            gate = (
                h1.astype(np.float32)
                * (1.0 / (1.0 + np.exp(-h1.astype(np.float32))))
                * h3.astype(np.float32)
            ).astype(np.float16)
            _ = k2.run_vec_fp16(gate)
        eval_ms = (time.perf_counter() - t0) * 1e3 / iters
    finally:
        k1.close()
        k3.close()
        k2.close()

    dim = int(w1.shape[1])
    hidden = int(w1.shape[0])
    gflops, tflops, util = flops_metrics(dim, hidden, eval_ms, peak_tflops)
    return BenchResult(
        dim, hidden, spatial, "linear_stack", hidden, compile_ms, eval_ms, gflops, tflops, util
    )


def bench_fused_mlp(
    bridge: ANEBridge,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
    x: np.ndarray,
    spatial: int,
    warmup: int,
    iters: int,
    peak_tflops: float,
) -> BenchResult:
    t0 = time.perf_counter()
    k = bridge.compile_fused_swiglu_ffn(w1, w3, w2, spatial=spatial)
    compile_ms = (time.perf_counter() - t0) * 1e3
    try:
        eval_ms = bench_kernel(k, x, warmup, iters)
    finally:
        k.close()

    dim = int(w1.shape[1])
    hidden = int(w1.shape[0])
    gflops, tflops, util = flops_metrics(dim, hidden, eval_ms, peak_tflops)
    total_mb = 3.0 * dim * hidden * 2.0 / (1024.0 * 1024.0)
    note = f"(weights~{total_mb:.1f}MB)"
    return BenchResult(
        dim, hidden, spatial, "fused_mlp", hidden, compile_ms, eval_ms, gflops, tflops, util, note=note
    )


def bench_fused_tiled(
    bridge: ANEBridge,
    w1: np.ndarray,
    w3: np.ndarray,
    w2: np.ndarray,
    x: np.ndarray,
    spatial: int,
    warmup: int,
    iters: int,
    peak_tflops: float,
    tile_hidden: int,
) -> BenchResult:
    t0 = time.perf_counter()
    k = bridge.compile_tiled_fused_swiglu_ffn(w1, w3, w2, hidden_tile=tile_hidden, spatial=spatial)
    compile_ms = (time.perf_counter() - t0) * 1e3
    try:
        eval_ms = bench_kernel(k, x, warmup, iters)
    finally:
        k.close()

    dim = int(w1.shape[1])
    hidden = int(w1.shape[0])
    gflops, tflops, util = flops_metrics(dim, hidden, eval_ms, peak_tflops)
    tile = min(tile_hidden, hidden)
    tile_mb = 3.0 * dim * tile * 2.0 / (1024.0 * 1024.0)
    note = f"(tile={tile_hidden}, tile_weights~{tile_mb:.1f}MB)"
    return BenchResult(
        dim,
        hidden,
        spatial,
        "fused_tiled",
        tile_hidden,
        compile_ms,
        eval_ms,
        gflops,
        tflops,
        util,
        note=note,
    )


def best_results_by_shape(results: list[BenchResult]) -> dict[tuple[int, int], BenchResult]:
    grouped: dict[tuple[int, int], BenchResult] = {}
    for res in results:
        key = (res.dim, res.hidden)
        prev = grouped.get(key)
        if prev is None or res.util_pct > prev.util_pct:
            grouped[key] = res
    return grouped


def write_jsonl(path: str, results: list[BenchResult]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    spatials = args.spatial_list if args.spatial_list else [args.spatial]
    tile_hidden_values = args.tile_hidden_list if args.tile_hidden_list else [args.tile_hidden]
    if any(s < 1 for s in spatials):
        raise RuntimeError("all spatial values must be >= 1")
    if any(t < 1 for t in tile_hidden_values):
        raise RuntimeError("all tile hidden values must be >= 1")
    if args.ane_sram_target_mb <= 0:
        raise RuntimeError("--ane-sram-target-mb must be > 0")
    if args.ane_tile_multiple < 1:
        raise RuntimeError("--ane-tile-multiple must be >= 1")
    if args.ane_min_hidden_tile < 1:
        raise RuntimeError("--ane-min-hidden-tile must be >= 1")

    bridge = ANEBridge(args.bridge_lib)
    results: List[BenchResult] = []

    for shape_str in args.shapes:
        dim, hidden = parse_shape(shape_str)
        print(f"\n=== shape dim={dim} hidden={hidden} ===")

        w1 = (rng.standard_normal((hidden, dim), dtype=np.float32) * 0.02).astype(np.float32)
        w3 = (rng.standard_normal((hidden, dim), dtype=np.float32) * 0.02).astype(np.float32)
        w2 = (rng.standard_normal((dim, hidden), dtype=np.float32) * 0.02).astype(np.float32)
        x = (rng.standard_normal((dim,), dtype=np.float32) * 0.2).astype(np.float16)

        for spatial in spatials:
            print(f"\n--- spatial={spatial} ---")
            if "linear_stack" in args.modes:
                try:
                    res = bench_linear_stack(
                        bridge, w1, w3, w2, x, spatial, args.warmup, args.iters, args.peak_tflops
                    )
                    results.append(res)
                    print(report(res))
                except Exception as exc:
                    print(f"linear_stack failed: {exc}")

            if "fused_mlp" in args.modes:
                try:
                    res = bench_fused_mlp(
                        bridge, w1, w3, w2, x, spatial, args.warmup, args.iters, args.peak_tflops
                    )
                    results.append(res)
                    print(report(res))
                except Exception as exc:
                    print(f"fused_mlp failed: {exc}")

            if "fused_tiled" in args.modes:
                for tile_hidden in tile_hidden_values:
                    try:
                        res = bench_fused_tiled(
                            bridge,
                            w1,
                            w3,
                            w2,
                            x,
                            spatial,
                            args.warmup,
                            args.iters,
                            args.peak_tflops,
                            tile_hidden,
                        )
                        results.append(res)
                        print(report(res))
                    except Exception as exc:
                        print(f"fused_tiled failed (tile={tile_hidden}): {exc}")

    print("\n=== summary ===")
    for res in results:
        print(report(res))

    if args.json_out:
        write_jsonl(args.json_out, results)
        print(f"\n[output] wrote JSONL rows to {args.json_out}")

    print("\n=== best config per shape ===")
    grouped = best_results_by_shape(results)
    for key in sorted(grouped.keys()):
        print(f"best {key[0]}:{key[1]} -> {report(grouped[key])}")

    print("\n=== policy recommendation per shape ===")
    for shape_str in args.shapes:
        dim, hidden = parse_shape(shape_str)
        choice = select_swiglu_shape(
            dim=dim,
            hidden=hidden,
            mode="mlp_tiled",
            requested_spatial=spatials[0],
            requested_hidden_tile=max(tile_hidden_values),
            auto_shape=args.ane_shape_policy == "auto",
            sram_target_mb=args.ane_sram_target_mb,
            tile_multiple=args.ane_tile_multiple,
            min_hidden_tile=args.ane_min_hidden_tile,
        )
        print(
            f"policy {dim}:{hidden} -> spatial={choice.spatial} tile={choice.hidden_tile} "
            f"tiles={choice.tile_count} tile_weights~{choice.tile_weight_mb:.1f}MB "
            f"(reason={choice.reason})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
