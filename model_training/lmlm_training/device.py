"""Conservative device and precision selection; never downloads or changes models."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DevicePlan:
    device: torch.device
    precision: str
    amp_enabled: bool
    reason: str

    def autocast(self):
        if not self.amp_enabled or self.precision == "fp32":
            return nullcontext()
        from .amp import autocast
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return autocast(device_type=self.device.type, dtype=dtype)

    def make_grad_scaler(self):
        if self.device.type != "cuda" or self.precision != "fp16" or not self.amp_enabled:
            return None
        # torch.amp.GradScaler(device, ...) is the newer API; PyTorch 2.1
        # exposes the CUDA-specific scaler instead.
        amp = getattr(torch, "amp", None)
        scaler_type = getattr(amp, "GradScaler", None) if amp is not None else None
        if scaler_type is not None:
            try:
                return scaler_type("cuda")
            except TypeError:
                pass
        return torch.cuda.amp.GradScaler()


def resolve_device_plan(device: str = "auto", precision: str = "auto",
                        *, allow_mps: bool = True) -> DevicePlan:
    """Resolve auto/cpu/cuda/mps and auto/fp32/fp16/bf16.

    Auto prefers CUDA, then Apple MPS, then CPU. Auto precision uses BF16 on
    BF16-capable CUDA, FP16 on other CUDA devices, and FP32 elsewhere. CPU/MPS
    mixed precision is opt-in so unsupported kernels are not selected silently.
    """
    if device == "auto":
        if torch.cuda.is_available():
            resolved = torch.device("cuda")
        elif allow_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved = torch.device("mps")
        else:
            resolved = torch.device("cpu")
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if resolved.type == "mps" and (
        not allow_mps or not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device type: {resolved.type}")

    if precision == "auto":
        if resolved.type == "cuda":
            bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            selected = "bf16" if bf16 else "fp16"
        else:
            selected = "fp32"
    else:
        selected = precision
    if selected not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be auto, fp32, fp16, or bf16")
    if selected == "fp16" and resolved.type != "cuda":
        raise ValueError("FP16 AMP is only enabled for CUDA by this runtime")
    if selected == "bf16" and resolved.type == "cuda":
        if not getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            raise RuntimeError("BF16 was requested but this CUDA device does not support it")
    if selected == "bf16" and resolved.type == "cpu" and not hasattr(torch, "autocast"):
        raise RuntimeError("CPU BF16 autocast is unavailable in this PyTorch build")
    return DevicePlan(resolved, selected, selected != "fp32",
                      f"selected {resolved.type}/{selected}")
