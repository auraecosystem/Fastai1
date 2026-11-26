"""Backward-compatible import shim for the historical mixed-case package path."""
from .lmlm_training import (
    BenchmarkResult, CheckpointManager, DevicePlan, TrainConfig, TrainResult,
    benchmark, evaluate, load_checkpoint, resolve_device_plan, train,
)

__all__ = [
    "BenchmarkResult", "CheckpointManager", "DevicePlan", "TrainConfig",
    "TrainResult", "benchmark", "evaluate", "load_checkpoint",
    "resolve_device_plan", "train",
]
