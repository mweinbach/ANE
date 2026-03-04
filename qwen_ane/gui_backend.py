#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import contextlib
import io
import json
import re
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

try:
    from transformers import AutoProcessor
except Exception:  # pragma: no cover - optional path
    AutoProcessor = None  # type: ignore[assignment]

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - optional path
    PILImage = None  # type: ignore[assignment]

from run_qwen35_4b_ane import (
    ANEBridge,
    OffloadManager,
    PowerMetrics,
    PowerMetricsSession,
    clamp_context_window,
    ensure_bridge_built,
    generate_from_inputs,
    load_qwen_model,
    patch_qwen_fused_mlps,
    patch_qwen_linears,
    resolve_kv_cache_dtype,
    resolve_prefill_device,
    model_runtime_meta,
    _encode_chat_messages,
)
from bench_ane_decode_kernels import (
    BenchResult,
    bench_fused_mlp,
    bench_fused_tiled,
    bench_linear_stack,
)


_EVENT_OUT = sys.__stdout__ if hasattr(sys, "__stdout__") and sys.__stdout__ is not None else sys.stdout


def emit(event: dict[str, Any]) -> None:
    _EVENT_OUT.write(json.dumps(event, ensure_ascii=False) + "\n")
    _EVENT_OUT.flush()


def perf_dict(perf) -> dict[str, Any]:
    return {
        "prompt_tokens": perf.prompt_tokens,
        "generated_tokens": perf.generated_tokens,
        "prefill_tokens": perf.prefill_tokens,
        "decode_tokens": perf.decode_tokens,
        "prefill_seconds": perf.prefill_seconds,
        "decode_seconds": perf.decode_seconds,
        "total_seconds": perf.total_seconds,
        "prefill_tps": perf.prefill_tps(),
        "decode_tps": perf.decode_tps(),
        "e2e_tps": perf.end_to_end_tps(),
    }


def power_dict(power: PowerMetrics | None) -> dict[str, Any] | None:
    if power is None:
        return None
    return {
        "soc_watts": power.soc_watts,
        "ane_watts": power.ane_watts,
        "cpu_watts": power.cpu_watts,
        "gpu_watts": power.gpu_watts,
        "sample_count": power.sample_count,
        "warning": power.warning,
    }


_POWER_LINE_RE = re.compile(
    r"(?i)\b([a-z0-9 _/\\-]*power[a-z0-9 _/\\-]*)\b[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mw|w)\b"
)
_THINK_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)


def _power_label_to_key(label: str) -> str | None:
    low = label.lower()
    if "ane" in low:
        return "ane_watts"
    if "cpu" in low:
        return "cpu_watts"
    if "gpu" in low:
        return "gpu_watts"
    if any(k in low for k in ("combined", "package", "soc", "total")):
        return "soc_watts"
    return None


def power_series_from_text(text: str, sample_rate_ms: int) -> list[dict[str, Any]]:
    if not text.strip():
        return []

    buckets: dict[str, list[float]] = {
        "soc_watts": [],
        "ane_watts": [],
        "cpu_watts": [],
        "gpu_watts": [],
    }

    for line in text.splitlines():
        m = _POWER_LINE_RE.search(line)
        if not m:
            continue
        key = _power_label_to_key(m.group(1))
        if key is None:
            continue
        raw = float(m.group(2))
        unit = m.group(3).lower()
        watts = raw / 1000.0 if unit == "mw" else raw
        buckets[key].append(watts)

    point_count = max((len(v) for v in buckets.values()), default=0)
    if point_count == 0:
        return []

    interval = max(sample_rate_ms, 1) / 1000.0
    out: list[dict[str, Any]] = []
    for idx in range(point_count):
        point: dict[str, Any] = {"index": idx, "t_sec": idx * interval}
        for key, arr in buckets.items():
            if idx < len(arr):
                point[key] = arr[idx]
        out.append(point)
    return out


def split_reasoning_answer(text: str) -> tuple[str | None, str]:
    raw = text or ""
    reasoning_parts = [m.strip() for m in _THINK_BLOCK_RE.findall(raw) if m.strip()]
    answer = _THINK_BLOCK_RE.sub("", raw).strip()
    reasoning = "\n\n".join(reasoning_parts).strip()
    return (reasoning if reasoning else None, answer)


def split_reasoning_answer_from_token_ids(
    tokenizer,
    token_ids: list[int],
    starts_in_reasoning: bool = False,
    fallback_text: str = "",
) -> tuple[str | None, str, str]:
    if not token_ids:
        reasoning, answer = split_reasoning_answer(fallback_text)
        return reasoning, answer, fallback_text

    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_start_id is None or think_end_id is None:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        reasoning, answer = split_reasoning_answer(text)
        return reasoning, answer, text

    reasoning_ids: list[int] = []
    answer_ids: list[int] = []
    mode = "reasoning" if starts_in_reasoning else "answer"
    saw_think = bool(starts_in_reasoning)

    for tid in token_ids:
        if tid == think_start_id:
            mode = "reasoning"
            saw_think = True
            continue
        if tid == think_end_id:
            mode = "answer"
            continue
        if mode == "reasoning":
            reasoning_ids.append(tid)
        else:
            answer_ids.append(tid)

    if not saw_think:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        reasoning, answer = split_reasoning_answer(text)
        return reasoning, answer, text

    reasoning = tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip()
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    full_visible = tokenizer.decode(token_ids, skip_special_tokens=True)
    return (reasoning if reasoning else None), answer, full_visible


def _extract_image_ref(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    image_ref: Any = item.get("image")
    if image_ref is None:
        image_ref = item.get("image_url")
        if isinstance(image_ref, dict):
            image_ref = image_ref.get("url")
    if isinstance(image_ref, str) and image_ref.strip():
        return image_ref.strip()
    return None


def _short_exc(exc: Exception) -> str:
    text = str(exc).strip()
    head = text.splitlines()[0] if text else exc.__class__.__name__
    return f"{exc.__class__.__name__}: {head}"


def _render_chat_prompt(
    tokenizer,
    template_driver,
    messages: list[dict[str, Any]],
    enable_thinking: bool | None,
) -> str:
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    try:
        return template_driver.apply_chat_template(messages, **template_kwargs)
    except TypeError:
        template_kwargs.pop("enable_thinking", None)
        return template_driver.apply_chat_template(messages, **template_kwargs)


def _load_vision_processor(model_id: str):
    if AutoProcessor is None:
        return None, "processor_unavailable", "AutoProcessor is unavailable in this transformers build"

    try:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        return proc, "auto_processor", None
    except Exception as first_exc:
        first_err = _short_exc(first_exc)

    try:
        from transformers import Qwen3VLProcessor

        proc = Qwen3VLProcessor.from_pretrained(model_id, trust_remote_code=True)
        return proc, "qwen3vl_processor_fallback", None
    except Exception as second_exc:
        second_err = _short_exc(second_exc)
        return None, "processor_load_failed", f"{first_err}; fallback failed: {second_err}"


def _vision_processor_preflight(tokenizer, processor) -> tuple[bool, str, str | None]:
    if processor is None:
        return False, "processor_missing", "no processor object available"
    if PILImage is None:
        return False, "pillow_missing", "Pillow is not installed"

    try:
        template_driver = processor if hasattr(processor, "apply_chat_template") else tokenizer
        probe_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "vision preflight"},
                    {"type": "image", "image": "file:///vision-preflight.png"},
                ],
            }
        ]
        prompt_text = _render_chat_prompt(
            tokenizer=tokenizer,
            template_driver=template_driver,
            messages=probe_messages,
            enable_thinking=False,
        )
        dummy_image = PILImage.new("RGB", (224, 224), color=(127, 127, 127))
        processed = processor(
            text=[prompt_text],
            images=[dummy_image],
            return_tensors="pt",
        )
        input_ids = processed.get("input_ids")
        if input_ids is None:
            raise RuntimeError("processor preflight did not produce input_ids")
        pixel_keys = sorted([key for key in processed.keys() if "pixel_values" in key])
        if not pixel_keys:
            raise RuntimeError("processor preflight did not produce pixel tensors")
        return True, f"ready({','.join(pixel_keys)})", None
    except Exception as exc:
        return False, "preflight_failed", _short_exc(exc)


def _message_has_image_content(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if _extract_image_ref(item) is not None:
                    return True
    return False


def _collect_image_refs(messages: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            ref = _extract_image_ref(item)
            if ref is not None:
                refs.append(ref)
    return refs


def _load_image_ref(ref: str):
    if PILImage is None:
        raise RuntimeError("Pillow is required for multimodal image preprocessing (pip install pillow)")

    parsed = urlparse(ref)
    if parsed.scheme in {"http", "https"}:
        with urlopen(ref, timeout=20) as resp:
            payload = resp.read()
        return PILImage.open(io.BytesIO(payload)).convert("RGB")

    if parsed.scheme == "file":
        path_str = unquote(parsed.path or "")
        if parsed.netloc:
            path_str = f"/{parsed.netloc}{path_str}"
        path = Path(path_str)
    else:
        path = Path(ref)

    if not path.exists():
        raise RuntimeError(f"image not found: {ref}")
    return PILImage.open(path).convert("RGB")


def _prepare_multimodal_inputs(
    tokenizer,
    processor,
    messages: list[dict[str, Any]],
    device: torch.device,
    enable_thinking: bool | None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    if processor is None:
        raise RuntimeError("multimodal message requested but no AutoProcessor is available")

    template_driver = processor if hasattr(processor, "apply_chat_template") else tokenizer
    prompt_text = _render_chat_prompt(
        tokenizer=tokenizer,
        template_driver=template_driver,
        messages=messages,
        enable_thinking=enable_thinking,
    )

    image_refs = _collect_image_refs(messages)
    if not image_refs:
        raise RuntimeError("multimodal preparation requested but no images were found")
    images = [_load_image_ref(ref) for ref in image_refs]

    processed = processor(
        text=[prompt_text],
        images=images,
        return_tensors="pt",
    )
    input_ids = processed.get("input_ids")
    if input_ids is None:
        raise RuntimeError("processor did not produce input_ids for multimodal request")
    input_ids = input_ids.to(device)
    attention_mask = processed.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    extra_model_inputs: dict[str, Any] = {}
    for key, value in processed.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if torch.is_tensor(value):
            extra_model_inputs[key] = value.to(device)
        else:
            extra_model_inputs[key] = value
    return input_ids, attention_mask, extra_model_inputs


def _make_synthetic_mlp_weights(dim: int, hidden: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    w1 = (rng.standard_normal((hidden, dim), dtype=np.float32) * 0.02).astype(np.float32)
    w3 = (rng.standard_normal((hidden, dim), dtype=np.float32) * 0.02).astype(np.float32)
    w2 = (rng.standard_normal((dim, hidden), dtype=np.float32) * 0.02).astype(np.float32)
    return w1, w3, w2


def _find_first_qwen_mlp(
    model: nn.Module, seed: int
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    for name, module in model.named_modules():
        gate = getattr(module, "gate_proj", None)
        up = getattr(module, "up_proj", None)
        down = getattr(module, "down_proj", None)
        if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
            continue

        dim = int(gate.in_features)
        hidden = int(gate.out_features)
        if dim < 1 or hidden < 1:
            continue

        w1 = gate.weight.detach().cpu().float().numpy()
        w3 = up.weight.detach().cpu().float().numpy()
        w2 = down.weight.detach().cpu().float().numpy()

        notes: list[str] = []
        if w1.shape == (dim, hidden) and w3.shape == (dim, hidden):
            w1 = np.ascontiguousarray(w1.T, dtype=np.float32)
            w3 = np.ascontiguousarray(w3.T, dtype=np.float32)
            notes.append("transposed gate/up")
        else:
            w1 = np.ascontiguousarray(w1, dtype=np.float32)
            w3 = np.ascontiguousarray(w3, dtype=np.float32)

        if w2.shape == (hidden, dim):
            w2 = np.ascontiguousarray(w2.T, dtype=np.float32)
            notes.append("transposed down")
        else:
            w2 = np.ascontiguousarray(w2, dtype=np.float32)

        if w1.shape == (hidden, dim) and w3.shape == (hidden, dim) and w2.shape == (dim, hidden):
            detail = ", ".join(notes) if notes else "model_weights"
            return name, w1, w3, w2, dim, hidden, detail

        packed_hint = (
            w1.shape[1] > 0
            and w2.shape[1] > 0
            and w1.shape[1] * 8 == dim
            and w2.shape[1] * 8 == hidden
        )
        source = "synthetic_from_features_packed" if packed_hint else "synthetic_from_features_shape_mismatch"
        synth_w1, synth_w3, synth_w2 = _make_synthetic_mlp_weights(dim=dim, hidden=hidden, seed=seed)
        return name, synth_w1, synth_w3, synth_w2, dim, hidden, source
    raise RuntimeError("unable to find qwen gate/up/down projection triplet for autotune")


def _int_list(req_value: Any, fallback: list[int]) -> list[int]:
    if req_value is None:
        return fallback
    if not isinstance(req_value, list) or not req_value:
        return fallback
    out: list[int] = []
    for v in req_value:
        try:
            iv = int(v)
        except Exception:
            continue
        if iv >= 1:
            out.append(iv)
    return out if out else fallback


def _autotune_decode_shape(
    bridge: ANEBridge,
    model: nn.Module,
    ane_mode: str,
    warmup: int,
    iters: int,
    spatial_values: list[int],
    tile_hidden_values: list[int],
    peak_tflops: float,
    seed: int,
) -> dict[str, Any]:
    module_name, w1, w3, w2, dim, hidden, weight_source = _find_first_qwen_mlp(model, seed=seed)
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((dim,), dtype=np.float32) * 0.2).astype(np.float16)
    results: list[BenchResult] = []
    failures: list[dict[str, Any]] = []

    mode_name = ane_mode
    if mode_name == "mlp_tiled":
        for spatial in spatial_values:
            for tile_hidden in tile_hidden_values:
                tile = max(1, min(int(tile_hidden), hidden))
                try:
                    row = bench_fused_tiled(
                        bridge=bridge,
                        w1=w1,
                        w3=w3,
                        w2=w2,
                        x=x,
                        spatial=int(spatial),
                        warmup=warmup,
                        iters=iters,
                        peak_tflops=peak_tflops,
                        tile_hidden=tile,
                    )
                    results.append(row)
                except Exception as exc:
                    failures.append(
                        {
                            "spatial": int(spatial),
                            "tile_hidden": tile,
                            "error": _short_exc(exc),
                        }
                    )
    elif mode_name == "mlp_fused":
        for spatial in spatial_values:
            try:
                row = bench_fused_mlp(
                    bridge=bridge,
                    w1=w1,
                    w3=w3,
                    w2=w2,
                    x=x,
                    spatial=int(spatial),
                    warmup=warmup,
                    iters=iters,
                    peak_tflops=peak_tflops,
                )
                results.append(row)
            except Exception as exc:
                failures.append({"spatial": int(spatial), "error": _short_exc(exc)})
    else:
        for spatial in spatial_values:
            try:
                row = bench_linear_stack(
                    bridge=bridge,
                    w1=w1,
                    w3=w3,
                    w2=w2,
                    x=x,
                    spatial=int(spatial),
                    warmup=warmup,
                    iters=iters,
                    peak_tflops=peak_tflops,
                )
                results.append(row)
            except Exception as exc:
                failures.append({"spatial": int(spatial), "error": _short_exc(exc)})

    if not results:
        detail = failures[0]["error"] if failures else "no benchmark rows produced"
        raise RuntimeError(f"autotune failed: {detail}")

    best = min(results, key=lambda row: row.eval_ms)
    recommended_hidden_tile = int(best.tile_hidden if mode_name == "mlp_tiled" else min(hidden, tile_hidden_values[-1]))
    return {
        "shape": {
            "module": module_name,
            "dim": dim,
            "hidden": hidden,
            "weight_source": weight_source,
        },
        "mode": mode_name,
        "warmup": warmup,
        "iters": iters,
        "candidates": [asdict(row) for row in results],
        "best": asdict(best),
        "recommended": {
            "ane_spatial": int(best.spatial),
            "ane_hidden_tile": int(recommended_hidden_tile),
        },
        "failures": failures,
        "bridge_compiles": bridge.compile_count(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Persistent JSON backend for ANE chat GUI")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    p.add_argument(
        "--bridge-lib",
        default=str(Path(__file__).resolve().parents[1] / "bridge" / "libane_bridge.dylib"),
    )
    p.add_argument("--prefill-device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--ane-mode", choices=["mlp_tiled", "mlp_fused", "linear"], default="mlp_tiled")
    p.add_argument("--ane-layers", type=int, default=12)
    p.add_argument("--ane-spatial", type=int, default=32)
    p.add_argument("--ane-hidden-tile", type=int, default=2048)
    p.add_argument("--ane-shape-policy", choices=["auto", "manual"], default="auto")
    p.add_argument("--ane-sram-target-mb", type=float, default=30.0)
    p.add_argument("--ane-tile-multiple", type=int, default=256)
    p.add_argument("--ane-min-hidden-tile", type=int, default=512)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--kv-cache-dtype", choices=["auto", "fp16", "bf16"], default="auto")
    p.add_argument("--powermetrics-sample-rate-ms", type=int, default=500)
    p.add_argument("--powermetrics-samplers", default="cpu_power,gpu_power,ane_power")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.ane_spatial < 1:
        raise RuntimeError("--ane-spatial must be >= 1")
    if args.ane_hidden_tile < 1:
        raise RuntimeError("--ane-hidden-tile must be >= 1")
    if args.ane_sram_target_mb <= 0:
        raise RuntimeError("--ane-sram-target-mb must be > 0")
    if args.ane_tile_multiple < 1:
        raise RuntimeError("--ane-tile-multiple must be >= 1")
    if args.ane_min_hidden_tile < 1:
        raise RuntimeError("--ane-min-hidden-tile must be >= 1")
    if args.powermetrics_sample_rate_ms < 10:
        raise RuntimeError("--powermetrics-sample-rate-ms must be >= 10")

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    lib_path = Path(args.bridge_lib)
    ensure_bridge_built(lib_path)
    prefill_device = resolve_prefill_device(args.prefill_device)

    with contextlib.redirect_stdout(sys.stderr):
        print(f"[backend] tokenizer: {args.model_id}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        processor, processor_source, processor_error = _load_vision_processor(args.model_id)
        vision_preflight_status = processor_source
        vision_preflight_error = processor_error
        if processor is not None:
            ok, preflight_status, preflight_error = _vision_processor_preflight(tokenizer, processor)
            vision_preflight_status = preflight_status
            vision_preflight_error = preflight_error
            if ok:
                print(f"[backend] processor: ready ({preflight_status})", flush=True)
            else:
                print(f"[backend] processor unavailable: {preflight_error}", flush=True)
                processor = None
        else:
            print(f"[backend] processor unavailable: {vision_preflight_error}", flush=True)
        print(f"[backend] model: {args.model_id} ({args.dtype})", flush=True)
        model = load_qwen_model(args.model_id, torch_dtype, prefill_device)
        runtime_meta = model_runtime_meta(model)
        for note in runtime_meta.get("notes", []):
            print(f"[backend] load note: {note}", flush=True)

    kv_cache_dtype = resolve_kv_cache_dtype(args.kv_cache_dtype, model)
    kv_cache_dtype_name = str(kv_cache_dtype).replace("torch.", "") if kv_cache_dtype is not None else "model-default"

    bridge = ANEBridge(lib_path)
    manager = OffloadManager()
    atexit.register(manager.close)

    with contextlib.redirect_stdout(sys.stderr):
        if args.ane_mode in {"mlp_fused", "mlp_tiled"}:
            stats = patch_qwen_fused_mlps(
                model,
                bridge,
                args.ane_layers,
                args.ane_spatial,
                args.ane_mode,
                args.ane_hidden_tile,
                args.ane_shape_policy == "auto",
                args.ane_sram_target_mb,
                args.ane_tile_multiple,
                args.ane_min_hidden_tile,
                manager,
            )
        else:
            stats = patch_qwen_linears(model, bridge, args.ane_layers, args.ane_spatial, manager)
        print(
            f"[backend] patch mode={args.ane_mode} wrapped {stats.patched}/{stats.attempted} modules",
            flush=True,
        )

    emit(
        {
            "event": "ready",
            "prefill_device": prefill_device.type,
            "ane_mode": args.ane_mode,
            "ane_layers": args.ane_layers,
            "ane_spatial": args.ane_spatial,
            "ane_hidden_tile": args.ane_hidden_tile,
            "ane_shape_policy": args.ane_shape_policy,
            "ane_sram_target_mb": args.ane_sram_target_mb,
            "bridge_compiles": bridge.compile_count(),
            "vision_processor_ready": processor is not None and PILImage is not None,
            "vision_processor_status": vision_preflight_status,
            "vision_processor_error": vision_preflight_error,
            "kv_cache_dtype_requested": args.kv_cache_dtype,
            "kv_cache_dtype_resolved": kv_cache_dtype_name,
            "model_quant_mode": runtime_meta.get("quant_mode"),
            "runtime_model_id": runtime_meta.get("runtime_model_id"),
        }
    )

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            req_type = req.get("type", "generate")
            if req_type == "shutdown":
                emit({"event": "shutdown", "id": req_id})
                break
            if req_type == "autotune":
                warmup = max(1, int(req.get("warmup", 4)))
                iters = max(1, int(req.get("iters", 30)))
                peak_tflops = float(req.get("peak_tflops", 15.8))
                seed = int(req.get("seed", 7))
                spatial_values = _int_list(req.get("spatial_values"), [16, 24, 32, 40, 48])
                tile_values = _int_list(req.get("tile_hidden_values"), [1024, 1536, 2048, 2560])
                with contextlib.redirect_stdout(sys.stderr):
                    print(
                        f"[backend] autotune start mode={args.ane_mode} warmup={warmup} iters={iters} "
                        f"spatials={spatial_values} tiles={tile_values}",
                        flush=True,
                    )
                    result = _autotune_decode_shape(
                        bridge=bridge,
                        model=model,
                        ane_mode=args.ane_mode,
                        warmup=warmup,
                        iters=iters,
                        spatial_values=spatial_values,
                        tile_hidden_values=tile_values,
                        peak_tflops=peak_tflops,
                        seed=seed,
                    )
                result["event"] = "autotune_result"
                result["id"] = req_id
                emit(result)
                continue

            if req_type != "generate":
                raise RuntimeError(f"unsupported request type: {req_type}")

            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                raise RuntimeError("messages must be a non-empty list")

            max_new_tokens = int(req.get("max_new_tokens", 128))
            temperature = float(req.get("temperature", 0.7))
            top_p = float(req.get("top_p", 0.9))
            top_k = int(req.get("top_k", 0))
            context_window = int(req.get("context_window", 0))
            with_power = bool(req.get("powermetrics", False))
            stream = bool(req.get("stream", False))
            reasoning_enabled = req.get("reasoning_enabled")
            if reasoning_enabled is not None:
                reasoning_enabled = bool(reasoning_enabled)

            if max_new_tokens < 1:
                raise RuntimeError("max_new_tokens must be >= 1")
            if top_k < 0:
                raise RuntimeError("top_k must be >= 0")
            if context_window < 0:
                raise RuntimeError("context_window must be >= 0")

            extra_model_inputs: dict[str, Any] | None = None
            multimodal_mode = "text"
            has_images = _message_has_image_content(messages)

            if has_images and processor is not None and PILImage is not None:
                try:
                    input_ids, attention_mask, extra_model_inputs = _prepare_multimodal_inputs(
                        tokenizer=tokenizer,
                        processor=processor,
                        messages=messages,
                        device=prefill_device,
                        enable_thinking=reasoning_enabled,
                    )
                    multimodal_mode = "vision_processor"
                except Exception as exc:
                    with contextlib.redirect_stdout(sys.stderr):
                        print(
                            f"[backend] multimodal preprocessor failed; using template tokens only: {exc}",
                            flush=True,
                        )
                    input_ids, attention_mask = _encode_chat_messages(
                        tokenizer,
                        messages,
                        prefill_device,
                        enable_thinking=reasoning_enabled,
                    )
                    multimodal_mode = "template_tokens_only"
            elif has_images:
                with contextlib.redirect_stdout(sys.stderr):
                    print(
                        f"[backend] multimodal input without ready vision processor; "
                        f"status={vision_preflight_status} error={vision_preflight_error}; using template tokens only",
                        flush=True,
                    )
                input_ids, attention_mask = _encode_chat_messages(
                    tokenizer,
                    messages,
                    prefill_device,
                    enable_thinking=reasoning_enabled,
                )
                multimodal_mode = "template_tokens_only"
            else:
                input_ids, attention_mask = _encode_chat_messages(
                    tokenizer,
                    messages,
                    prefill_device,
                    enable_thinking=reasoning_enabled,
                )

            if context_window > 0 and not has_images:
                input_ids, attention_mask = clamp_context_window(input_ids, attention_mask, context_window)
            elif context_window > 0 and has_images:
                with contextlib.redirect_stdout(sys.stderr):
                    print(
                        "[backend] context_window ignored for multimodal request to preserve vision-token alignment",
                        flush=True,
                    )

            power_session = PowerMetricsSession(
                enabled=with_power,
                sample_rate_ms=args.powermetrics_sample_rate_ms,
                samplers=args.powermetrics_samplers,
            )
            power_session.start()
            streamed_ids: list[int] = []
            streamed_text = ""

            if stream:
                emit({"event": "response_start", "id": req_id})

            def on_token(token_id: int, generated_count: int) -> None:
                nonlocal streamed_text
                streamed_ids.append(int(token_id))
                decoded = tokenizer.decode(streamed_ids, skip_special_tokens=True)
                if decoded.startswith(streamed_text):
                    delta = decoded[len(streamed_text) :]
                else:
                    delta = decoded
                streamed_text = decoded
                reasoning, answer, _visible = split_reasoning_answer_from_token_ids(
                    tokenizer,
                    streamed_ids,
                    starts_in_reasoning=bool(reasoning_enabled),
                )
                if not stream:
                    return
                emit(
                    {
                        "event": "response_chunk",
                        "id": req_id,
                        "delta": delta,
                        "text": streamed_text,
                        "reasoning": reasoning,
                        "answer": answer,
                        "generated_tokens": generated_count,
                    }
                )

            with contextlib.redirect_stdout(sys.stderr):
                _full_text, new_text, perf = generate_from_inputs(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    model_inputs=extra_model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    kv_cache_dtype=kv_cache_dtype,
                    on_token=on_token,
                )
            power, power_text = power_session.stop_with_text()
            power_series = power_series_from_text(power_text, args.powermetrics_sample_rate_ms)
            reply = new_text.strip() if new_text.strip() else new_text
            reasoning, answer, visible = split_reasoning_answer_from_token_ids(
                tokenizer,
                streamed_ids,
                starts_in_reasoning=bool(reasoning_enabled),
                fallback_text=reply,
            )
            if not answer:
                if reasoning:
                    answer = ""
                else:
                    answer = visible.strip() or reply
            compiled = sum(1 for w in manager.wrappers if w.kernel is not None)
            assistant = answer if answer else ("" if reasoning else reply)

            if stream and streamed_text != reply:
                if reply.startswith(streamed_text):
                    delta = reply[len(streamed_text) :]
                else:
                    delta = reply
                streamed_text = reply
                emit(
                    {
                        "event": "response_chunk",
                        "id": req_id,
                        "delta": delta,
                        "text": streamed_text,
                        "reasoning": reasoning,
                        "answer": answer,
                        "generated_tokens": perf.generated_tokens,
                    }
                )

            emit(
                {
                    "event": "response",
                    "id": req_id,
                    "assistant": assistant,
                    "raw_assistant": reply,
                    "reasoning": reasoning,
                    "answer": answer,
                    "perf": perf_dict(perf),
                    "power": power_dict(power),
                    "power_series": power_series,
                    "bridge_compiles": bridge.compile_count(),
                    "ane_kernels_compiled": compiled,
                    "multimodal_mode": multimodal_mode,
                    "kv_cache_dtype": kv_cache_dtype_name,
                    "model_quant_mode": runtime_meta.get("quant_mode"),
                }
            )
        except Exception as exc:
            emit(
                {
                    "event": "error",
                    "id": req_id,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    manager.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
