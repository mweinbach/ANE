# Qwen ANE Swift Port (Runtime + GUI)

This package now includes:

- ANE bridge bindings (MIL compile/eval via private ANE APIs)
- MIL generators for:
  - linear conv
  - fused SwiGLU FFN
  - tiled fused SwiGLU FFN
- ANE kernel execution wrappers (`runFP16` / `runVecFP16`)
- `powermetrics` integration and parser (SoC/ANE/CPU/GPU watts)
- perf telemetry output (prefill/decode/e2e tok/s style metrics)
- native SwiftUI chat GUI (`qwen-ane-gui`) that drives the Python Qwen backend

## Build

```bash
cd swift_qwen_ane
swift build -c debug
```

Binary:

```bash
./.build/arm64-apple-macosx/debug/qwen-ane-swift
```

GUI binary:

```bash
./.build/arm64-apple-macosx/debug/qwen-ane-gui
```

## Usage

Current command:

```bash
qwen-ane-swift bench [options]
```

Example:

```bash
./.build/arm64-apple-macosx/debug/qwen-ane-swift bench \
  --shape 2560:9216 \
  --mode fused_tiled \
  --tile-hidden 2048 \
  --spatial 32 \
  --warmup 5 \
  --iters 20
```

Power telemetry (requires root):

```bash
sudo ./.build/arm64-apple-macosx/debug/qwen-ane-swift bench \
  --shape 2560:9216 \
  --mode fused_tiled \
  --tile-hidden 2048 \
  --powermetrics
```

## GUI Usage

Run from the repository (or a subdirectory where `qwen_ane/gui_backend.py` can be found by walking parent dirs):

```bash
cd /Users/mweinbach/Projects/ANE/swift_qwen_ane
swift run -c debug qwen-ane-gui
```

In the GUI:

- Left: chat transcript and input box.
- Assistant responses stream token-by-token (enable/disable via `Stream Responses`).
- Qwen 3.5 chat-template `enable_thinking` is controlled by `Enable Reasoning`.
- `<think>...</think>` output is parsed into a dedicated reasoning panel instead of being mixed into final answer text.
- You can attach images from the composer (`Add Images`) or drag-and-drop image files onto the composer area.
- User turns with images are sent as multimodal chat-template content (`text` + `image` items).
- Right sidebar controls:
  - model catalog presets with local-cache detection, one-click download, and "best available" selection
    - `mlx-community/Qwen3.5-4B-mxfp4`
    - `mlx-community/Qwen3.5-2B-6bit`
  - `Probe Selected` warm-start check that launches a short-lived backend with the selected model/runtime settings and reports ready/failure details
  - sampling (`context_window`, `max_new_tokens`, `temperature`, `top_p`, `top_k`)
  - backend/runtime options (`model id/path`, prefill device, ANE mode/layers/spatial/tile, shape policy, SRAM target, dtype)
  - `Auto-Tune ANE` benchmark button (benchmarks candidate decode shapes, applies best `spatial/tile`, and restarts backend)
  - optional `Auto-Tune On Startup` toggle
  - `powermetrics` controls and samplers
- Stats cards show:
  - prefill/decode/e2e tok/s
  - prompt/generated token counts
  - SoC/ANE/CPU/GPU watts from `powermetrics` when available
  - a power-over-time chart (ANE + SoC watts)
  - bridge/kernel compile counters

When `Capture Powermetrics` is enabled and backend starts, the app requests administrator access using a GUI prompt and launches the backend under `sudo -A`.

If you toggle `Capture Powermetrics` after launch, click `Restart Backend` to apply privilege changes.

Default ANE shape policy is `auto` with `ANE SRAM Target MB = 30`, which maps the Qwen3.5-4B MLP shape (`dim=2560, hidden=9216`) to `spatial=32, hidden_tile=2048` on this setup.

## Note

The GUI is native SwiftUI, but inference is currently executed by the Python backend (`qwen_ane/gui_backend.py`) so it can reuse the existing model/tokenizer/runtime stack.

Current backend uses text-generation model loading (`AutoModelForCausalLM`) and a multimodal `AutoProcessor` path with a startup preflight check.

For full image understanding, point `Model ID / Path` to a vision-capable Qwen model and ensure `torchvision` is installed (required by `Qwen3VLVideoProcessor` in current transformers builds). If no working processor is available, the app falls back to chat-template placeholders (`template_tokens_only` mode), and visual grounding will be limited.

Model downloads in the catalog use Python `huggingface_hub.snapshot_download` and store in the standard HF cache under `~/.cache/huggingface/hub`.
