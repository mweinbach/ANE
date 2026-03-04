#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import os
import re
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ane_bridge import ANEBridge, ANEKernel
from shape_policy import select_swiglu_shape


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_bridge_lib() -> Path:
    return _repo_root() / "bridge" / "libane_bridge.dylib"


def ensure_bridge_built(lib_path: Path) -> None:
    if lib_path.exists():
        return
    bridge_dir = _repo_root() / "bridge"
    print(f"[build] {lib_path} missing, building in {bridge_dir}")
    subprocess.run(["make"], cwd=bridge_dir, check=True)
    if not lib_path.exists():
        raise RuntimeError(f"bridge build completed but {lib_path} still missing")


_LAYER_RE = re.compile(r"(^|\.)layers\.(\d+)(\.|$)")


def layer_index_from_name(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    if not m:
        return None
    return int(m.group(2))


def get_parent_and_attr(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


@dataclass
class PatchStats:
    attempted: int = 0
    patched: int = 0


@dataclass
class TurnPerfMetrics:
    prompt_tokens: int
    generated_tokens: int
    prefill_tokens: int
    decode_tokens: int
    prefill_seconds: float
    decode_seconds: float
    total_seconds: float

    def prefill_tps(self) -> float:
        if self.prefill_seconds <= 0:
            return 0.0
        return self.prefill_tokens / self.prefill_seconds

    def decode_tps(self) -> float:
        if self.decode_seconds <= 0:
            return 0.0
        return self.decode_tokens / self.decode_seconds

    def end_to_end_tps(self) -> float:
        if self.total_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.total_seconds


@dataclass
class PowerMetrics:
    soc_watts: Optional[float] = None
    ane_watts: Optional[float] = None
    cpu_watts: Optional[float] = None
    gpu_watts: Optional[float] = None
    sample_count: int = 0
    warning: Optional[str] = None


class PowerMetricsSession:
    def __init__(self, enabled: bool, sample_rate_ms: int, samplers: str):
        self.enabled = enabled
        self.sample_rate_ms = sample_rate_ms
        self.samplers = samplers
        self._proc: Optional[subprocess.Popen] = None
        self._tmp_path: Optional[Path] = None
        self._warning: Optional[str] = None

    def start(self) -> None:
        if not self.enabled:
            return
        if os.geteuid() != 0:
            self._warning = "powermetrics requires sudo/root; skipping power capture"
            return

        tmp = tempfile.NamedTemporaryFile(prefix="powermetrics_", suffix=".txt", delete=False)
        tmp.close()
        self._tmp_path = Path(tmp.name)

        cmd = [
            "powermetrics",
            "--samplers",
            self.samplers,
            "--sample-rate",
            str(self.sample_rate_ms),
            "--sample-count",
            "-1",
            "--output-file",
            str(self._tmp_path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            self._warning = f"failed to start powermetrics: {exc}"
            self._cleanup_tmp()
            return

        # If it exits immediately, capture a useful error.
        time.sleep(0.08)
        if self._proc.poll() is not None:
            err = ""
            if self._proc.stderr is not None:
                err = self._proc.stderr.read().strip()
            self._warning = err or "powermetrics exited immediately"
            self._proc = None
            self._cleanup_tmp()

    def stop(self) -> Optional[PowerMetrics]:
        metrics, _text = self.stop_with_text()
        return metrics

    def stop_with_text(self) -> Tuple[Optional[PowerMetrics], str]:
        if not self.enabled:
            return None, ""

        if self._proc is not None:
            try:
                self._proc.send_signal(signal.SIGINT)
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
                except Exception:
                    pass

            if self._proc.stderr is not None:
                err = self._proc.stderr.read().strip()
                if err and self._warning is None:
                    self._warning = err

        text = ""
        if self._tmp_path is not None and self._tmp_path.exists():
            try:
                text = self._tmp_path.read_text(errors="ignore")
            except Exception as exc:
                if self._warning is None:
                    self._warning = f"failed reading powermetrics output: {exc}"
        self._cleanup_tmp()

        metrics = _parse_powermetrics_text(text)
        metrics.warning = _merge_warnings(metrics.warning, self._warning)
        return metrics, text

    def _cleanup_tmp(self) -> None:
        if self._tmp_path is not None:
            try:
                self._tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._tmp_path = None


class ANELinear(nn.Module):
    def __init__(self, base: nn.Linear, bridge: ANEBridge, name: str, spatial: int):
        super().__init__()
        self.base = base
        self.bridge = bridge
        self.name = name
        self.spatial = spatial
        self.kernel: Optional[ANEKernel] = None
        self.compile_error: Optional[str] = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def _compile_if_needed(self) -> None:
        if self.kernel is not None or self.compile_error is not None:
            return
        try:
            w = self.base.weight.detach().cpu().float().numpy()
            self.kernel = self.bridge.compile_linear(w, spatial=self.spatial)
            print(
                f"[ane] compiled {self.name} [{self.out_features}, {self.in_features}] "
                f"(spatial={self.spatial}, compiles={self.bridge.compile_count()})"
            )
        except Exception as exc:  # pragma: no cover - runtime fallback path
            self.compile_error = str(exc)
            print(f"[ane] fallback {self.name}: {self.compile_error}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep prefill and non-single-token shapes on CPU torch path.
        if x.ndim < 3 or x.shape[0] != 1 or x.shape[1] != 1:
            return self.base(x)

        self._compile_if_needed()
        if self.kernel is None:
            return self.base(x)

        x16 = x.detach().to(dtype=torch.float16, device="cpu").contiguous()
        flat = x16.view(-1)
        np_in = flat.numpy()
        np_out = self.kernel.run_vec_fp16(np_in)

        out = torch.from_numpy(np_out.copy()).view(1, 1, self.out_features)
        out = out.to(dtype=x.dtype, device=x.device)

        if self.base.bias is not None:
            out = out + self.base.bias.to(dtype=out.dtype, device=out.device).view(1, 1, -1)
        return out

    def close(self) -> None:
        if self.kernel is not None:
            self.kernel.close()
            self.kernel = None


class ANEFusedQwenMLP(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        bridge: ANEBridge,
        name: str,
        spatial: int,
        mode: str,
        hidden_tile: int,
        auto_shape: bool,
        sram_target_mb: float,
        tile_multiple: int,
        min_hidden_tile: int,
    ):
        super().__init__()
        self.base = base
        self.bridge = bridge
        self.name = name
        self.spatial = spatial
        self.mode = mode
        self.hidden_tile = hidden_tile
        self.auto_shape = auto_shape
        self.sram_target_mb = sram_target_mb
        self.tile_multiple = tile_multiple
        self.min_hidden_tile = min_hidden_tile
        self.kernel: Optional[ANEKernel] = None
        self.compile_error: Optional[str] = None

    def _compile_if_needed(self) -> None:
        if self.kernel is not None or self.compile_error is not None:
            return
        try:
            gate = getattr(self.base, "gate_proj", None)
            up = getattr(self.base, "up_proj", None)
            down = getattr(self.base, "down_proj", None)
            if not isinstance(gate, nn.Linear) or not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
                raise RuntimeError("module is missing gate_proj/up_proj/down_proj Linear layers")
            if gate.bias is not None or up.bias is not None:
                raise RuntimeError("fused kernel path requires bias-free gate/up projections")

            w1 = gate.weight.detach().cpu().float().numpy()
            w3 = up.weight.detach().cpu().float().numpy()
            w2 = down.weight.detach().cpu().float().numpy()

            choice = select_swiglu_shape(
                dim=gate.in_features,
                hidden=gate.out_features,
                mode=self.mode,
                requested_spatial=self.spatial,
                requested_hidden_tile=self.hidden_tile,
                auto_shape=self.auto_shape,
                sram_target_mb=self.sram_target_mb,
                tile_multiple=self.tile_multiple,
                min_hidden_tile=self.min_hidden_tile,
            )
            chosen_spatial = choice.spatial
            chosen_hidden_tile = choice.hidden_tile

            if self.mode == "mlp_tiled":
                self.kernel = self.bridge.compile_tiled_fused_swiglu_ffn(
                    w1,
                    w3,
                    w2,
                    hidden_tile=chosen_hidden_tile,
                    spatial=chosen_spatial,
                )
                mode_desc = (
                    f"tiled-mlp(hidden_tile={chosen_hidden_tile}, tiles={choice.tile_count}, "
                    f"tile_weights~{choice.tile_weight_mb:.1f}MB/{self.sram_target_mb:.1f}MB)"
                )
            else:
                self.kernel = self.bridge.compile_fused_swiglu_ffn(w1, w3, w2, spatial=chosen_spatial)
                mode_desc = f"fused-mlp(weights~{choice.total_weight_mb:.1f}MB)"

            print(
                f"[ane] compiled {mode_desc} {self.name} "
                f"[dim={gate.in_features}, hidden={gate.out_features}] "
                f"(spatial={chosen_spatial}, policy={choice.reason}, compiles={self.bridge.compile_count()})"
            )
        except Exception as exc:  # pragma: no cover - runtime fallback path
            self.compile_error = str(exc)
            print(f"[ane] fallback {self.name}: {self.compile_error}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep prefill and non-single-token shapes on torch path.
        if x.ndim < 3 or x.shape[0] != 1 or x.shape[1] != 1:
            return self.base(x)

        self._compile_if_needed()
        if self.kernel is None:
            return self.base(x)

        x16 = x.detach().to(dtype=torch.float16, device="cpu").contiguous()
        np_in = x16.view(-1).numpy()
        np_out = self.kernel.run_vec_fp16(np_in)
        out = torch.from_numpy(np_out.copy()).view(1, 1, -1).to(dtype=x.dtype, device=x.device)

        down = getattr(self.base, "down_proj", None)
        if isinstance(down, nn.Linear) and down.bias is not None:
            out = out + down.bias.to(dtype=out.dtype, device=out.device).view(1, 1, -1)
        return out

    def close(self) -> None:
        if self.kernel is not None:
            self.kernel.close()
            self.kernel = None


class OffloadManager:
    def __init__(self) -> None:
        self.wrappers: List[nn.Module] = []

    def register(self, wrapper: nn.Module) -> None:
        self.wrappers.append(wrapper)

    def close(self) -> None:
        for w in self.wrappers:
            w.close()


def patch_qwen_linears(
    model: nn.Module,
    bridge: ANEBridge,
    max_layers: int,
    spatial: int,
    manager: OffloadManager,
) -> PatchStats:
    stats = PatchStats()

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        layer_idx = layer_index_from_name(name)
        if layer_idx is None or layer_idx >= max_layers:
            continue

        stats.attempted += 1

        parent, attr = get_parent_and_attr(model, name)
        wrapper = ANELinear(module, bridge, name, spatial=spatial)
        setattr(parent, attr, wrapper)
        manager.register(wrapper)
        stats.patched += 1

    return stats


def patch_qwen_fused_mlps(
    model: nn.Module,
    bridge: ANEBridge,
    max_layers: int,
    spatial: int,
    mode: str,
    hidden_tile: int,
    auto_shape: bool,
    sram_target_mb: float,
    tile_multiple: int,
    min_hidden_tile: int,
    manager: OffloadManager,
) -> PatchStats:
    stats = PatchStats()

    for name, module in list(model.named_modules()):
        layer_idx = layer_index_from_name(name)
        if layer_idx is None or layer_idx >= max_layers:
            continue

        has_qwen_mlp_shape = (
            hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
            and hasattr(module, "down_proj")
            and isinstance(getattr(module, "gate_proj"), nn.Linear)
            and isinstance(getattr(module, "up_proj"), nn.Linear)
            and isinstance(getattr(module, "down_proj"), nn.Linear)
        )
        if not has_qwen_mlp_shape:
            continue

        stats.attempted += 1
        parent, attr = get_parent_and_attr(model, name)
        wrapper = ANEFusedQwenMLP(
            module,
            bridge,
            name,
            spatial=spatial,
            mode=mode,
            hidden_tile=hidden_tile,
            auto_shape=auto_shape,
            sram_target_mb=sram_target_mb,
            tile_multiple=tile_multiple,
            min_hidden_tile=min_hidden_tile,
        )
        setattr(parent, attr, wrapper)
        manager.register(wrapper)
        stats.patched += 1

    return stats


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, top_k: int = 0) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)

    scaled = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled.shape[-1])
        topk_vals, _ = torch.topk(scaled, k=k, dim=-1)
        kth = topk_vals[..., -1, None]
        scaled = torch.where(scaled < kth, torch.full_like(scaled, -torch.inf), scaled)
    probs = torch.softmax(scaled, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        to_remove = cumulative > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = False
        sorted_probs[to_remove] = 0
        denom = sorted_probs.sum(dim=-1, keepdim=True)
        sorted_probs = sorted_probs / torch.clamp(denom, min=1e-12)
        sampled = torch.multinomial(sorted_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled)
    else:
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token.squeeze(-1)


def _merge_warnings(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a and b:
        return f"{a}; {b}"
    return a or b


def _avg_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def _parse_powermetrics_text(text: str) -> PowerMetrics:
    if not text.strip():
        return PowerMetrics()

    line_re = re.compile(
        r"(?i)\b([a-z0-9 _/\\-]*power[a-z0-9 _/\\-]*)\b[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*(mw|w)\b"
    )
    ane_vals: list[float] = []
    cpu_vals: list[float] = []
    gpu_vals: list[float] = []
    soc_vals: list[float] = []

    for line in text.splitlines():
        m = line_re.search(line)
        if not m:
            continue
        label = m.group(1).strip().lower()
        raw = float(m.group(2))
        unit = m.group(3).lower()
        watts = raw / 1000.0 if unit == "mw" else raw

        if "ane" in label:
            ane_vals.append(watts)
            continue
        if "cpu" in label:
            cpu_vals.append(watts)
            continue
        if "gpu" in label:
            gpu_vals.append(watts)
            continue
        if any(k in label for k in ("combined", "package", "soc", "total")):
            soc_vals.append(watts)

    soc = _avg_or_none(soc_vals)
    cpu = _avg_or_none(cpu_vals)
    gpu = _avg_or_none(gpu_vals)
    ane = _avg_or_none(ane_vals)
    if soc is None and any(v is not None for v in (cpu, gpu, ane)):
        soc = float(sum(v for v in (cpu, gpu, ane) if v is not None))

    return PowerMetrics(
        soc_watts=soc,
        ane_watts=ane,
        cpu_watts=cpu,
        gpu_watts=gpu,
        sample_count=max(len(ane_vals), len(cpu_vals), len(gpu_vals), len(soc_vals)),
    )


def load_qwen_model(model_id: str, dtype: torch.dtype, device: torch.device) -> nn.Module:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        extra = ""
        msg = str(exc)
        if "model type `qwen3_5`" in msg or "model type 'qwen3_5'" in msg:
            extra = (
                " This environment does not include Qwen3.5 architecture support yet. "
                "Try: pip install --upgrade transformers "
                "or pip install git+https://github.com/huggingface/transformers.git."
            )
        raise RuntimeError(
            f"AutoModelForCausalLM failed for {model_id}. "
            "Use a recent transformers build and verify model access. "
            f"Original error: {exc}.{extra}"
        ) from exc

    model.to(device)
    model.eval()
    return model


def resolve_prefill_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this machine/runtime")
        return torch.device("mps")
    raise ValueError(f"unsupported prefill device: {name}")


def _encode_prompt(tokenizer, prompt: str, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _encode_chat_messages(
    tokenizer,
    messages: list[dict[str, Any]],
    device: torch.device,
    enable_thinking: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            try:
                prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                # Backward compatibility for older tokenizers that do not accept enable_thinking.
                kwargs.pop("enable_thinking", None)
                prompt_text = tokenizer.apply_chat_template(messages, **kwargs)
            return _encode_prompt(tokenizer, prompt_text, device)
        except Exception:
            pass

    def fallback_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
                        continue

                    image_ref: Any = item.get("image")
                    if image_ref is None:
                        image_ref = item.get("image_url")
                        if isinstance(image_ref, dict):
                            image_ref = image_ref.get("url")
                    if image_ref is not None:
                        if isinstance(image_ref, str) and image_ref:
                            parts.append(f"[image:{image_ref}]")
                        else:
                            parts.append("[image]")
                        continue

                    parts.append(str(item))
                else:
                    parts.append(str(item))
            return " ".join(p for p in parts if p).strip()
        return str(content)

    rendered = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = fallback_content_text(msg.get("content", ""))
        rendered.append(f"{role}: {content}")
    rendered.append("assistant:")
    return _encode_prompt(tokenizer, "\n".join(rendered), device)


def clamp_context_window(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    context_window: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if context_window <= 0:
        return input_ids, attention_mask
    if input_ids.shape[1] <= context_window:
        return input_ids, attention_mask
    input_ids = input_ids[:, -context_window:]
    if attention_mask is not None:
        attention_mask = attention_mask[:, -context_window:]
    return input_ids, attention_mask


def generate_from_inputs(
    model: nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    model_inputs: Optional[dict[str, Any]] = None,
    on_token: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, str, TurnPerfMetrics]:
    prompt_tokens = int(input_ids.shape[1])
    generated = input_ids.clone()
    prefill_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
    }
    if model_inputs:
        prefill_kwargs.update(model_inputs)

    with torch.inference_mode():
        # Prefill (usually seq > 1) runs on the configured prefill device.
        t_prefill0 = time.perf_counter()
        out = model(**prefill_kwargs)
        prefill_seconds = time.perf_counter() - t_prefill0
        past = out.past_key_values
        next_token = sample_next_token(out.logits[:, -1, :], temperature, top_p, top_k=top_k)

        generated_tokens = 0
        decode_tokens = 0
        t_decode0 = time.perf_counter()

        while generated_tokens < max_new_tokens:
            generated = torch.cat([generated, next_token.view(1, 1)], dim=1)
            generated_tokens += 1
            if on_token is not None:
                try:
                    on_token(int(next_token.item()), generated_tokens)
                except Exception:
                    pass
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
            if generated_tokens >= max_new_tokens:
                break

            out = model(input_ids=next_token.view(1, 1), past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = sample_next_token(out.logits[:, -1, :], temperature, top_p, top_k=top_k)
            decode_tokens += 1

        decode_seconds = time.perf_counter() - t_decode0

    total_seconds = prefill_seconds + decode_seconds
    new_token_ids = generated[0, prompt_tokens:]
    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    new_text = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    perf = TurnPerfMetrics(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prefill_tokens=prompt_tokens,
        decode_tokens=decode_tokens,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        total_seconds=total_seconds,
    )
    return full_text, new_text, perf


def print_turn_metrics(perf: TurnPerfMetrics, power: Optional[PowerMetrics]) -> None:
    print(
        "[perf] "
        f"prompt_tokens={perf.prompt_tokens} "
        f"generated_tokens={perf.generated_tokens} "
        f"prefill={perf.prefill_seconds:.3f}s ({perf.prefill_tps():.2f} tok/s) "
        f"decode={perf.decode_seconds:.3f}s ({perf.decode_tps():.2f} tok/s) "
        f"e2e={perf.total_seconds:.3f}s ({perf.end_to_end_tps():.2f} tok/s)"
    )
    if power is None:
        return
    if power.warning:
        print(f"[power] warning: {power.warning}")
    vals = []
    if power.soc_watts is not None:
        vals.append(f"SoC={power.soc_watts:.2f}W")
    if power.ane_watts is not None:
        vals.append(f"ANE={power.ane_watts:.2f}W")
    if power.cpu_watts is not None:
        vals.append(f"CPU={power.cpu_watts:.2f}W")
    if power.gpu_watts is not None:
        vals.append(f"GPU={power.gpu_watts:.2f}W")
    if vals:
        sample_suffix = f" samples={power.sample_count}" if power.sample_count > 0 else ""
        print(f"[power] {' '.join(vals)}{sample_suffix}")


def run_one_shot(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    prefill_device: torch.device,
) -> None:
    input_ids, attention_mask = _encode_prompt(tokenizer, args.prompt, prefill_device)
    input_ids, attention_mask = clamp_context_window(input_ids, attention_mask, args.context_window)
    power_session = PowerMetricsSession(
        enabled=args.powermetrics,
        sample_rate_ms=args.powermetrics_sample_rate_ms,
        samplers=args.powermetrics_samplers,
    )
    power_session.start()
    full_text, _new_text, perf = generate_from_inputs(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    power = power_session.stop()
    print_turn_metrics(perf, power)
    print("\n=== OUTPUT ===\n")
    print(full_text)


def run_chat_repl(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
    prefill_device: torch.device,
) -> None:
    print("[chat] Interactive mode. Commands: /exit, /quit, /reset")
    messages: list[dict[str, str]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    while True:
        try:
            user_text = input("\nuser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] exiting")
            return

        if not user_text:
            continue
        lowered = user_text.lower()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            print("[chat] exiting")
            return
        if lowered == "/reset":
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            print("[chat] history reset")
            continue

        messages.append({"role": "user", "content": user_text})
        input_ids, attention_mask = _encode_chat_messages(tokenizer, messages, prefill_device)
        input_ids, attention_mask = clamp_context_window(input_ids, attention_mask, args.context_window)

        power_session = PowerMetricsSession(
            enabled=args.powermetrics,
            sample_rate_ms=args.powermetrics_sample_rate_ms,
            samplers=args.powermetrics_samplers,
        )
        power_session.start()
        _full_text, new_text, perf = generate_from_inputs(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        power = power_session.stop()
        reply = new_text.strip() if new_text.strip() else new_text
        print(f"\nassistant> {reply}")
        print_turn_metrics(perf, power)
        messages.append({"role": "assistant", "content": reply})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Qwen 4B with GPU prefill + ANE decode offload")
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B", help="HF model id")
    p.add_argument("--prompt", default="Write a short haiku about Apple Neural Engine.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0, help="Top-k sampling cutoff (0 disables)")
    p.add_argument(
        "--context-window",
        type=int,
        default=0,
        help="Maximum prompt context tokens to keep before decode (0 disables truncation)",
    )
    p.add_argument(
        "--ane-layers",
        type=int,
        default=12,
        help="How many transformer layers to wrap with ANE decode kernels",
    )
    p.add_argument(
        "--ane-mode",
        choices=["mlp_tiled", "mlp_fused", "linear"],
        default="mlp_tiled",
        help="Decode offload strategy: tiled fused MLP, fused MLP, or per-Linear kernels",
    )
    p.add_argument(
        "--dtype",
        choices=["fp16", "bf16"],
        default="fp16",
        help="Load dtype for the HF model",
    )
    p.add_argument(
        "--bridge-lib",
        default=str(_default_bridge_lib()),
        help="Path to libane_bridge.dylib",
    )
    p.add_argument(
        "--prefill-device",
        choices=["mps", "cpu"],
        default="mps",
        help="Device for non-ANE work (prefill and fallback ops)",
    )
    p.add_argument(
        "--ane-spatial",
        type=int,
        default=32,
        help="Spatial size used inside decode ANE kernels (must be >= 1)",
    )
    p.add_argument(
        "--ane-hidden-tile",
        type=int,
        default=2048,
        help="Hidden chunk size for --ane-mode mlp_tiled (keeps each tile in a smaller SRAM-friendly working set)",
    )
    p.add_argument(
        "--ane-shape-policy",
        choices=["auto", "manual"],
        default="auto",
        help="Kernel shape policy for fused MLP path",
    )
    p.add_argument(
        "--ane-sram-target-mb",
        type=float,
        default=30.0,
        help="Target SRAM-sized tile working-set budget used by --ane-shape-policy auto",
    )
    p.add_argument(
        "--ane-tile-multiple",
        type=int,
        default=256,
        help="Alignment multiple for auto hidden tile selection",
    )
    p.add_argument(
        "--ane-min-hidden-tile",
        type=int,
        default=512,
        help="Minimum hidden tile for auto policy when hidden is large enough",
    )
    p.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat loop mode",
    )
    p.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt for --chat mode",
    )
    p.add_argument(
        "--powermetrics",
        action="store_true",
        help="Capture SoC/ANE/CPU/GPU power using powermetrics during each turn (requires sudo/root)",
    )
    p.add_argument(
        "--powermetrics-sample-rate-ms",
        type=int,
        default=500,
        help="powermetrics sample interval in milliseconds",
    )
    p.add_argument(
        "--powermetrics-samplers",
        default="cpu_power,gpu_power,ane_power",
        help="powermetrics sampler list",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    torch_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    lib_path = Path(args.bridge_lib)
    ensure_bridge_built(lib_path)
    prefill_device = resolve_prefill_device(args.prefill_device)

    print(f"[load] tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    print(f"[load] model: {args.model_id} ({args.dtype})")
    t0 = time.time()
    model = load_qwen_model(args.model_id, torch_dtype, prefill_device)
    print(f"[load] done in {time.time() - t0:.1f}s")

    bridge = ANEBridge(lib_path)
    manager = OffloadManager()
    atexit.register(manager.close)

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
    if args.top_k < 0:
        raise RuntimeError("--top-k must be >= 0")
    if args.context_window < 0:
        raise RuntimeError("--context-window must be >= 0")
    if args.powermetrics_sample_rate_ms < 10:
        raise RuntimeError("--powermetrics-sample-rate-ms must be >= 10")

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
        f"[patch] mode={args.ane_mode} wrapped {stats.patched}/{stats.attempted} modules "
        f"for layers [0..{max(args.ane_layers - 1, 0)}]"
    )
    print(
        f"[run] prefill_device={prefill_device.type}, "
        f"decode_backend=ANE/{args.ane_mode}, ane_spatial={args.ane_spatial}, "
        f"ane_hidden_tile={args.ane_hidden_tile}, ane_shape_policy={args.ane_shape_policy}, "
        f"ane_sram_target_mb={args.ane_sram_target_mb:.1f}"
    )
    if args.powermetrics and os.geteuid() != 0:
        print("[power] powermetrics requested but process is not root; power capture will be skipped")

    try:
        if args.chat:
            run_chat_repl(model, tokenizer, args, prefill_device)
        else:
            print("[run] generating...")
            run_one_shot(model, tokenizer, args, prefill_device)
        compiled = sum(1 for w in manager.wrappers if w.kernel is not None)
        print(f"[run] ane_kernels_compiled={compiled}, bridge_compiles={bridge.compile_count()}")
    finally:
        manager.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
