"""PyTorch-backed, evidence-based introspection for LMLM training.

This module reports observable model/runtime/training state. It does not claim
consciousness, subjective experience, or access to hidden internal thoughts.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from time import monotonic
from typing import Any, Mapping

import torch
from torch import nn

__all__ = [
    "TensorHealth",
    "ModelSnapshot",
    "TrainingSnapshot",
    "ModelIntrospector",
    "TrainingAwareness",
    "SelfAwareness",
]


def _number(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected a scalar tensor.")
        return float(value.detach().cpu().item())
    return float(value)


@dataclass(frozen=True)
class TensorHealth:
    """Numerical health summary for a tensor, without retaining the tensor."""

    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    finite: bool
    finite_fraction: float
    mean: float | None
    std: float | None
    min: float | None
    max: float | None


@dataclass(frozen=True)
class ModelSnapshot:
    """Observable architecture, parameter, mode, and device information."""

    class_name: str
    training: bool
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    parameter_bytes: int
    buffers: int
    module_count: int
    module_types: dict[str, int]
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    gradients_present: int
    gradients_non_finite: int


@dataclass(frozen=True)
class TrainingSnapshot:
    """Training counters and recent measured loss behavior."""

    epoch: int
    global_step: int
    samples_seen: int
    latest_loss: float | None
    mean_recent_loss: float | None
    best_loss: float | None
    loss_is_finite: bool | None
    non_finite_loss_count: int
    elapsed_seconds: float
    latest_metrics: dict[str, float] = field(default_factory=dict)


class ModelIntrospector:
    """Inspect public PyTorch module state and tensor health."""

    @staticmethod
    def inspect(model: nn.Module) -> ModelSnapshot:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")

        parameters = list(model.parameters())
        buffers = list(model.buffers())
        module_types: dict[str, int] = {}
        for module in model.modules():
            name = type(module).__name__
            module_types[name] = module_types.get(name, 0) + 1

        devices = sorted({str(t.device) for t in (*parameters, *buffers)})
        dtypes = sorted({str(t.dtype) for t in (*parameters, *buffers)})
        gradients = [p.grad for p in parameters if p.grad is not None]

        return ModelSnapshot(
            class_name=f"{type(model).__module__}.{type(model).__qualname__}",
            training=model.training,
            total_parameters=sum(p.numel() for p in parameters),
            trainable_parameters=sum(
                p.numel() for p in parameters if p.requires_grad
            ),
            frozen_parameters=sum(
                p.numel() for p in parameters if not p.requires_grad
            ),
            parameter_bytes=sum(
                p.numel() * p.element_size() for p in parameters
            ),
            buffers=sum(t.numel() for t in buffers),
            module_count=sum(module_types.values()),
            module_types=module_types,
            devices=tuple(devices),
            dtypes=tuple(dtypes),
            gradients_present=len(gradients),
            gradients_non_finite=sum(
                int(not bool(torch.isfinite(g).all().item())) for g in gradients
            ),
        )

    @staticmethod
    def inspect_tensor(tensor: torch.Tensor) -> TensorHealth:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")

        detached = tensor.detach()
        if detached.is_complex():
            values = detached.abs().to(dtype=torch.float64)
        elif detached.is_floating_point():
            values = detached.to(dtype=torch.float64)
        else:
            values = detached.to(dtype=torch.float64)

        if values.numel() == 0:
            finite_fraction = 1.0
            finite = True
            mean = std = minimum = maximum = None
        else:
            mask = torch.isfinite(values)
            finite_fraction = float(mask.float().mean().cpu().item())
            finite = bool(mask.all().cpu().item())
            safe = values[mask]
            if safe.numel():
                mean = float(safe.mean().cpu().item())
                std = float(safe.std(unbiased=False).cpu().item())
                minimum = float(safe.min().cpu().item())
                maximum = float(safe.max().cpu().item())
            else:
                mean = std = minimum = maximum = None

        return TensorHealth(
            shape=tuple(detached.shape),
            dtype=str(detached.dtype),
            device=str(detached.device),
            numel=detached.numel(),
            finite=finite,
            finite_fraction=finite_fraction,
            mean=mean,
            std=std,
            min=minimum,
            max=maximum,
        )


class TrainingAwareness:
    """Track explicit training events and bounded recent loss history."""

    def __init__(self, history_size: int = 100) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        self.epoch = 0
        self.global_step = 0
        self.samples_seen = 0
        self.latest_loss: float | None = None
        self.best_loss: float | None = None
        self.loss_is_finite: bool | None = None
        self.non_finite_loss_count = 0
        self.latest_metrics: dict[str, float] = {}
        self._losses: deque[float] = deque(maxlen=history_size)
        self._started = monotonic()

    def start_epoch(self, epoch: int | None = None) -> None:
        next_epoch = self.epoch + 1 if epoch is None else int(epoch)
        if next_epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = next_epoch

    def record_step(
        self,
        loss: float | torch.Tensor,
        *,
        batch_size: int = 0,
        metrics: Mapping[str, float | torch.Tensor] | None = None,
    ) -> TrainingSnapshot:
        if batch_size < 0:
            raise ValueError("batch_size must be non-negative")
        value = _number(loss)
        self.global_step += 1
        self.samples_seen += int(batch_size)
        self.latest_loss = value
        self.loss_is_finite = bool(torch.isfinite(torch.tensor(value)).item())
        if self.loss_is_finite:
            self._losses.append(value)
            self.best_loss = value if self.best_loss is None else min(self.best_loss, value)
        else:
            self.non_finite_loss_count += 1

        if metrics:
            self.latest_metrics = {str(k): _number(v) for k, v in metrics.items()}
        return self.snapshot()

    def snapshot(self) -> TrainingSnapshot:
        recent_mean = (
            sum(self._losses) / len(self._losses) if self._losses else None
        )
        return TrainingSnapshot(
            epoch=self.epoch,
            global_step=self.global_step,
            samples_seen=self.samples_seen,
            latest_loss=self.latest_loss,
            mean_recent_loss=recent_mean,
            best_loss=self.best_loss,
            loss_is_finite=self.loss_is_finite,
            non_finite_loss_count=self.non_finite_loss_count,
            elapsed_seconds=max(0.0, monotonic() - self._started),
            latest_metrics=dict(self.latest_metrics),
        )


class SelfAwareness:
    """Combine model introspection with explicitly recorded training evidence.

    The object does not hook into forward passes or alter gradients. Call
    record_step() from the trainer when a step has completed.
    """

    def __init__(self, model: nn.Module, *, history_size: int = 100) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        self.model = model
        self.introspector = ModelIntrospector()
        self.training = TrainingAwareness(history_size=history_size)

    def inspect(self) -> dict[str, Any]:
        """Return a JSON-friendly snapshot of known model and training state."""
        model_snapshot = self.introspector.inspect(self.model)
        training_snapshot = self.training.snapshot()
        return {
            "scope": "observable_runtime_state",
            "model": asdict(model_snapshot),
            "training": asdict(training_snapshot),
            "limitations": [
                "Reports measured state; it does not establish consciousness.",
                "Capabilities require task-specific evaluation evidence.",
                "Confidence requires a calibrated, task-appropriate estimator.",
            ],
        }

    def start_epoch(self, epoch: int | None = None) -> None:
        self.training.start_epoch(epoch)

    def record_step(
        self,
        loss: float | torch.Tensor,
        *,
        batch_size: int = 0,
        metrics: Mapping[str, float | torch.Tensor] | None = None,
    ) -> TrainingSnapshot:
        return self.training.record_step(
            loss, batch_size=batch_size, metrics=metrics
        )

    def inspect_tensor(self, tensor: torch.Tensor) -> TensorHealth:
        return self.introspector.inspect_tensor(tensor)
