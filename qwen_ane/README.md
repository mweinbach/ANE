# Qwen 3.5 4B ANE Runner (Hybrid)

This adds a **hybrid runner** for `Qwen/Qwen3.5-4B`:

- GPU prefill on `mps` (or CPU fallback)
- ANE offload for decode (`seq=1`) using either:
  - tiled fused Qwen MLP kernels (`--ane-mode mlp_tiled`, default)
  - fused Qwen MLP kernels (`--ane-mode mlp_fused`)
  - per-Linear kernels (`--ane-mode linear`, baseline)
- Uses this repo's private-API bridge (`bridge/libane_bridge.dylib`)

It is designed to be a practical starting point, not a full production runtime.

## Why Hybrid

This ANE path uses baked-weight MIL kernels. On current code/hardware there is a practical compile budget per process, so compiling every linear in a 4B model at once is risky. The script defaults to partial layer offload (`--ane-layers 12`) to stay within budget.

## Prerequisites

- macOS 15+ on Apple Silicon
- Python 3.10+
- Build tools (`xcrun clang`, `make`)
- Python packages:

```bash
pip install torch torchvision transformers safetensors sentencepiece pillow
```

## Build Bridge

```bash
cd bridge
make
```

This produces `bridge/libane_bridge.dylib`.

## Run

From repo root:

```bash
python3 qwen_ane/run_qwen35_4b_ane.py \
  --model-id Qwen/Qwen3.5-4B \
  --prompt "Explain ANE kernel fusion in two paragraphs." \
  --max-new-tokens 64 \
  --prefill-device mps \
  --ane-mode mlp_tiled \
  --ane-hidden-tile 2048 \
  --ane-spatial 64 \
  --ane-layers 12
```

Useful flags:

- `--ane-layers N`: number of transformer layers to wrap for ANE offload
- `--ane-mode mlp_tiled|mlp_fused|linear`: decode kernel strategy
- `--ane-hidden-tile N`: hidden chunk size used by `mlp_tiled` (default `2048`)
- `--prefill-device mps|cpu`: device for prefill and all non-ANE ops
- `--ane-spatial N`: internal spatial packing for decode kernels (default `32`, commonly `32`/`64` are stable)
- `--ane-shape-policy auto|manual`: per-layer shape selection policy (default `auto`)
- `--ane-sram-target-mb N`: SRAM-target budget for auto tile selection (default `30`)
- `--ane-tile-multiple N`: hidden tile alignment for auto policy (default `256`)
- `--dtype fp16|bf16`: model load dtype
- `--temperature` and `--top-p`: sampling controls
- `--bridge-lib /path/to/libane_bridge.dylib`: custom bridge path
- `--chat`: interactive chat loop (supports `/reset` and `/quit`)
- `--system-prompt "...": optional system prompt for chat mode`
- `--powermetrics`: collect SoC/CPU/GPU/ANE power (requires root)
- `--powermetrics-sample-rate-ms N`: sample interval for powermetrics

## Verification

### 1) Fused-kernel correctness test

Compares ANE fused SwiGLU-FFN output vs CPU reference math.

```bash
python3 qwen_ane/tests/test_fused_mlp_correctness.py \
  --dim 256 --hidden 768 --spatial 32 --trials 8
```

### 2) Tiled fused-kernel correctness test

```bash
python3 qwen_ane/tests/test_tiled_fused_mlp_correctness.py \
  --dim 2560 --hidden 9216 --tile-hidden 2048 --spatial 32 --trials 6
```

### 3) Utilization mapping benchmark

Benchmarks per-shape decode kernels and reports latency, GFLOPS, and estimated ANE utilization.

```bash
python3 qwen_ane/bench_ane_decode_kernels.py \
  --shapes 768:2048 1024:4096 2560:9216 \
  --spatial-list 16 24 32 40 48 64 \
  --tile-hidden-list 1024 1536 2048 2560 \
  --ane-shape-policy auto \
  --ane-sram-target-mb 30 \
  --iters 50
```

This runs:
- `linear_stack`: three ANE linears + host SwiGLU math
- `fused_mlp`: single fused ANE MLP kernel
- `fused_tiled`: tiled fused kernel that keeps each hidden chunk in a smaller working set

Use the same shapes as your target models to map practical utilization and identify SRAM/dispatch bottlenecks.
The benchmark prints both measured best configs and policy recommendations per shape, and can emit JSONL with `--json-out`.

### 4) Shape-policy unit tests (no ANE hardware required)

```bash
python3 -m unittest discover qwen_ane/tests -p "test_shape_policy.py" -v
python3 -m unittest discover qwen_ane/tests -p "test_bench_recommendations.py" -v
python3 -m unittest discover qwen_ane/tests -p "test_auto_shape_wrapper.py" -v
```

## Chat + Telemetry

Interactive chat interface:

```bash
python3 qwen_ane/run_qwen35_4b_ane.py \
  --model-id /Users/mweinbach/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/manual \
  --prefill-device mps \
  --ane-mode mlp_tiled \
  --ane-hidden-tile 2048 \
  --ane-spatial 32 \
  --ane-layers 12 \
  --chat \
  --max-new-tokens 128
```

Per turn it prints:
- prefill tokens/sec
- decode tokens/sec
- end-to-end tokens/sec

Power capture with `powermetrics` (requires `sudo`):

```bash
sudo python3 qwen_ane/run_qwen35_4b_ane.py \
  --model-id /Users/mweinbach/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/manual \
  --prefill-device mps \
  --ane-mode mlp_tiled \
  --ane-hidden-tile 2048 \
  --ane-spatial 32 \
  --ane-layers 12 \
  --chat \
  --powermetrics \
  --powermetrics-sample-rate-ms 500
```

If not run as root, the runner keeps working and prints a warning that power capture was skipped.

## Notes

- For any module that fails ANE compile, the wrapper automatically falls back to CPU (`torch.nn.Linear`) for that module.
- Decode-time ANE path currently targets single-token steps (`batch=1, seq=1`), which is the common autoregressive loop.
- MLX itself currently targets CPU/GPU; ANE decode here is provided by this repo's private ANE bridge, not by MLX runtime.
- If your local `transformers` build cannot load `Qwen/Qwen3.5-4B`, try upgrading `transformers` and verifying model access. You can also test the same runner with `Qwen/Qwen3-4B`.
