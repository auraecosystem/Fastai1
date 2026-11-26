"""Small reproducible inference benchmark; reports measurements, not performance claims."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import torch
from torch import nn
from .device import resolve_device_plan

@dataclass
class BenchmarkResult:
    device: str
    precision: str
    batch_size: int
    warmup_iterations: int
    measured_iterations: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    samples_per_second: float
    peak_memory_bytes: int | None

    def to_dict(self):
        return asdict(self)

def benchmark(model: nn.Module, inputs, *, device: str = "auto", precision: str = "auto",
              warmup: int = 5, iterations: int = 30) -> BenchmarkResult:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations >= 1")
    plan = resolve_device_plan(device, precision)
    model = model.to(plan.device).eval()
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(plan.device)
        batch_size = int(inputs.shape[0]) if inputs.ndim else 1
    else:
        raise TypeError("benchmark inputs must be a tensor")
    if plan.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(plan.device)
    def run():
        with torch.inference_mode(), plan.autocast():
            model(inputs)
        if plan.device.type == "cuda":
            torch.cuda.synchronize(plan.device)
    for _ in range(warmup):
        run()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        run()
        timings.append((time.perf_counter() - start) * 1000.0)
    values = torch.tensor(timings, dtype=torch.float64)
    mean_ms = float(values.mean())
    p50 = float(torch.quantile(values, 0.50))
    p95 = float(torch.quantile(values, 0.95))
    total_samples = batch_size * iterations
    peak = torch.cuda.max_memory_allocated(plan.device) if plan.device.type == "cuda" else None
    return BenchmarkResult(str(plan.device), plan.precision, batch_size, warmup, iterations,
                           mean_ms, p50, p95, total_samples / (sum(timings) / 1000.0), peak)
