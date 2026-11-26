"""Conservative device and precision selection; never downloads or changes models."""
from __future__ import annotations
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
            from contextlib import nullcontext
            return nullcontext()
        dtype = torch.float16 if self.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype)

    def make_grad_scaler(self):
        if self.device.type == "cuda" and self.precision == "fp16" and self.amp_enabled:
            return torch.amp.GradScaler("cuda")
        return None

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
    if selected == "bf16" and resolved.type == "cpu":
        if not hasattr(torch, "autocast"):
            raise RuntimeError("CPU BF16 autocast is unavailable in this PyTorch build")
    amp = selected != "fp32"
    reason = f"selected {resolved.type}/{selected}"
    return DevicePlan(resolved, selected, amp, reason)
