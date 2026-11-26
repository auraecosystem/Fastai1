"""Canonical lowercase import path for the LMLM PyTorch training adapter.

The implementation remains in the historical Lmlm_training directory for
backward compatibility with existing repository consumers.
"""
from model_training.Lmlm_training import (
    BenchmarkResult, CheckpointManager, DevicePlan, TrainConfig, TrainResult,
    benchmark, evaluate, load_checkpoint, resolve_device_plan, train,
)

__all__ = [
    "BenchmarkResult", "CheckpointManager", "DevicePlan", "TrainConfig",
    "TrainResult", "benchmark", "evaluate", "load_checkpoint",
    "resolve_device_plan", "train",
]
