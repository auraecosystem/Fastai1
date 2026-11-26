from pathlib import Path
import textwrap, zipfile, json, os

root = Path("/mnt/data/modern_training_extensions")
(root / "training_extensions").mkdir(parents=True, exist_ok=True)
(root / "tests").mkdir(exist_ok=True)

files = {}

files["training_extensions/__init__.py"] = r'''
"""Modern, framework-light PyTorch training extensions.

This package replaces the uploaded fastai v1 module with explicit PyTorch APIs.
It does not monkey-patch torch or fastai classes.
"""
from .augmentation import mixup_batch, MixupResult
from .callbacks import freeze_batchnorm_stats, clip_gradients, GradientAccumulator
from .engine import fit, FitResult, EpochResult
from .interpretation import (
    Interpretation, ClassificationInterpretation,
    MultiLabelClassificationInterpretation,
)
from .precision import autocast_context, make_grad_scaler, PrecisionConfig
from .schedulers import (
    build_one_cycle_scheduler, build_flat_cosine_scheduler,
    find_learning_rate, LRFindResult,
)

__all__ = [
    "mixup_batch", "MixupResult", "freeze_batchnorm_stats",
    "clip_gradients", "GradientAccumulator", "fit", "FitResult",
    "EpochResult", "Interpretation", "ClassificationInterpretation",
    "MultiLabelClassificationInterpretation", "autocast_context",
    "make_grad_scaler", "PrecisionConfig", "build_one_cycle_scheduler",
    "build_flat_cosine_scheduler", "find_learning_rate", "LRFindResult",
]
'''

files["training_extensions/schedulers.py"] = r'''
"""Learning-rate schedules and a state-safe learning-rate finder."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence
import copy
import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR, CosineAnnealingLR


def _validate_optimizer(optimizer: Optimizer) -> None:
    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer")


def build_one_cycle_scheduler(
    optimizer: Optimizer,
    *,
    max_lr: float | Sequence[float],
    total_steps: Optional[int] = None,
    epochs: Optional[int] = None,
    steps_per_epoch: Optional[int] = None,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    three_phase: bool = False,
) -> OneCycleLR:
    """Create PyTorch's OneCycleLR with explicit step-count validation.

    Call ``scheduler.step()`` once after each successful optimizer update.
    For gradient accumulation, total_steps must count optimizer updates,
    not micro-batches.
    """
    _validate_optimizer(optimizer)
    if total_steps is None:
        if epochs is None or steps_per_epoch is None:
            raise ValueError("provide total_steps or both epochs and steps_per_epoch")
        if epochs <= 0 or steps_per_epoch <= 0:
            raise ValueError("epochs and steps_per_epoch must be positive")
        total_steps = epochs * steps_per_epoch
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < pct_start < 1.0:
        raise ValueError("pct_start must be between 0 and 1")
    if div_factor <= 0 or final_div_factor <= 0:
        raise ValueError("div_factor values must be positive")
    return OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=pct_start, anneal_strategy=anneal_strategy,
        div_factor=div_factor, final_div_factor=final_div_factor,
        three_phase=three_phase,
    )


def build_flat_cosine_scheduler(
    optimizer: Optimizer,
    *,
    total_steps: int,
    start_flat_fraction: float = 0.72,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Hold the initial LR flat, then cosine-decay it to ``min_lr_ratio``.

    This scheduler is stepped per optimizer update.
    """
    _validate_optimizer(optimizer)
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= start_flat_fraction < 1.0:
        raise ValueError("start_flat_fraction must be in [0, 1)")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    flat_steps = int(total_steps * start_flat_fraction)
    decay_steps = max(1, total_steps - flat_steps)

    def multiplier(step: int) -> float:
        if step < flat_steps:
            return 1.0
        progress = min(1.0, max(0.0, (step - flat_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


@dataclass
class LRFindResult:
    learning_rates: list[float]
    losses: list[float]
    suggested_lr: Optional[float]

    def as_dict(self) -> dict:
        return {
            "learning_rates": list(self.learning_rates),
            "losses": list(self.losses),
            "suggested_lr": self.suggested_lr,
        }


def find_learning_rate(
    model: nn.Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    batches: Iterable,
    *,
    device: torch.device | str,
    start_lr: float = 1e-7,
    end_lr: float = 10.0,
    num_steps: int = 100,
    diverge_factor: float = 4.0,
    step_fn: Optional[Callable] = None,
) -> LRFindResult:
    """Run an exponential LR range test and restore model/optimizer state.

    ``batches`` must be re-iterable (e.g. a DataLoader), not a one-shot
    generator. Batch items are ``(inputs, targets)``. ``step_fn`` may perform
    custom forward/loss computation and must return a scalar loss.
    The suggested LR is the LR at the steepest finite loss descent.
    """
    _validate_optimizer(optimizer)
    if start_lr <= 0 or end_lr <= start_lr:
        raise ValueError("require 0 < start_lr < end_lr")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    if diverge_factor <= 1:
        raise ValueError("diverge_factor must be greater than 1")
    if iter(batches) is batches:
        raise TypeError("batches must be re-iterable; pass a DataLoader or list")

    model_state = copy.deepcopy(model.state_dict())
    optim_state = copy.deepcopy(optimizer.state_dict())
    was_training = model.training
    device = torch.device(device)
    rates, losses = [], []
    best_loss = float("inf")
    iterator = iter(batches)
    multiplier = (end_lr / start_lr) ** (1 / (num_steps - 1))
    lr = start_lr

    try:
        model.to(device)
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = start_lr

        for _ in range(num_steps):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(batches)
                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    break

            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (step_fn(model, inputs, targets) if step_fn
                    else loss_fn(model(inputs), targets))
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor")
            value = float(loss.detach().item())
            rates.append(lr)
            losses.append(value)
            if not math.isfinite(value):
                break
            if value < best_loss:
                best_loss = value
            if len(losses) > 1 and value > diverge_factor * best_loss:
                break
            loss.backward()
            optimizer.step()
            lr *= multiplier
            for group in optimizer.param_groups:
                group["lr"] = lr
    finally:
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optim_state)
        model.train(was_training)

    suggested = None
    finite_pairs = [(i, x) for i, x in enumerate(losses) if math.isfinite(x)]
    if len(finite_pairs) >= 2:
        gradients = []
        for (i0, y0), (i1, y1) in zip(finite_pairs, finite_pairs[1:]):
            gradients.append(((y1 - y0) / max(1, i1 - i0), i1))
        _, idx = min(gradients, key=lambda pair: pair[0])
        if idx < len(rates):
            suggested = rates[idx]
    return LRFindResult(rates, losses, suggested)
'''

files["training_extensions/precision.py"] = r'''
"""Device-aware automatic mixed precision helpers."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PrecisionConfig:
    """Precision policy for modern PyTorch.

    ``mode`` may be ``"fp32"``, ``"fp16"``, or ``"bf16"``.
    FP16 scaling is normally useful on CUDA; BF16 generally does not need it.
    """
    mode: str = "fp32"
    enabled: bool = True

    def __post_init__(self):
        if self.mode not in {"fp32", "fp16", "bf16"}:
            raise ValueError("mode must be 'fp32', 'fp16', or 'bf16'")


def autocast_context(device: torch.device | str, config: PrecisionConfig):
    """Return an autocast context appropriate for the requested device."""
    device = torch.device(device)
    if not config.enabled or config.mode == "fp32":
        return nullcontext()
    if config.mode == "fp16" and device.type != "cuda":
        raise ValueError("fp16 autocast is supported here only on CUDA; use bf16 or fp32")
    dtype = torch.float16 if config.mode == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)


def make_grad_scaler(device: torch.device | str, config: PrecisionConfig):
    """Create a GradScaler when CUDA FP16 is requested; otherwise return None."""
    device = torch.device(device)
    if config.enabled and config.mode == "fp16" and device.type == "cuda":
        # Current API; supported by recent PyTorch releases.
        return torch.amp.GradScaler("cuda")
    return None
'''

files["training_extensions/augmentation.py"] = r'''
"""Batch-level data augmentation."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class MixupResult:
    inputs: torch.Tensor
    targets_a: torch.Tensor
    targets_b: torch.Tensor
    lam: float


def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.4,
    generator: torch.Generator | None = None,
) -> MixupResult:
    """Apply MixUp to a batch, returning both targets and the mixing weight.

    Supports classification targets as class indices or dense/multi-label
    targets. The caller computes ``lam * loss(pred, targets_a) + ...``.
    """
    if inputs.ndim < 2 or inputs.shape[0] < 2:
        raise ValueError("inputs must contain at least two samples")
    if targets.shape[0] != inputs.shape[0]:
        raise ValueError("inputs and targets must have the same batch size")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    concentration = torch.tensor([alpha], device=inputs.device, dtype=torch.float32)
    lam = float(torch.distributions.Beta(concentration, concentration).sample().item())
    permutation = torch.randperm(inputs.shape[0], device=inputs.device, generator=generator)
    mixed = lam * inputs + (1.0 - lam) * inputs[permutation]
    return MixupResult(mixed, targets, targets[permutation], lam)
'''

files["training_extensions/callbacks.py"] = r'''
"""Small, composable training utilities replacing fastai callbacks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import math
import torch
from torch import nn


def freeze_batchnorm_stats(model: nn.Module) -> None:
    """Put BatchNorm modules in eval mode without freezing affine parameters.

    Call after ``model.train()`` (e.g. at the beginning of each epoch).
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def clip_gradients(
    model: nn.Module,
    max_norm: float,
    *,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip gradients of trainable parameters and return the total norm."""
    if max_norm <= 0 or not math.isfinite(max_norm):
        raise ValueError("max_norm must be a finite positive number")
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(
        params, max_norm, norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
    )


@dataclass
class GradientAccumulator:
    """Correct sample-weighted gradient accumulation.

    Use ``backward(loss * batch_size)`` for mean-reduced losses, then call
    ``step`` at an accumulation boundary. Gradients are normalized by the
    actual number of samples in the window, including a short final window.
    Call ``zero_grad`` before beginning a new window.
    """
    accumulation_steps: int = 1
    drop_last: bool = False

    def __post_init__(self):
        if self.accumulation_steps < 1:
            raise ValueError("accumulation_steps must be >= 1")
        self.micro_batches = 0
        self.samples = 0

    def backward(self, loss: torch.Tensor, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        (loss * batch_size).backward()
        self.micro_batches += 1
        self.samples += batch_size

    def ready(self) -> bool:
        return self.micro_batches >= self.accumulation_steps

    def step(self, optimizer, model: nn.Module, *, scaler=None) -> bool:
        """Normalize and update. Returns False if there is no pending work."""
        if self.micro_batches == 0:
            return False
        if self.micro_batches < self.accumulation_steps and self.drop_last:
            optimizer.zero_grad(set_to_none=True)
            self.reset()
            return False
        if scaler is not None:
            scaler.unscale_(optimizer)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(self.samples)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        optimizer.zero_grad(set_to_none=True)
        self.reset()
        return True

    def reset(self):
        self.micro_batches = 0
        self.samples = 0
'''

files["training_extensions/interpretation.py"] = r'''
"""Prediction interpretation utilities for classification tasks."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Interpretation:
    predictions: torch.Tensor
    targets: torch.Tensor
    losses: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.predictions.shape[0] != self.targets.shape[0]:
            raise ValueError("predictions and targets must have equal sample counts")
        if self.losses is not None and self.losses.shape[0] != self.targets.shape[0]:
            raise ValueError("losses must have one value per sample")

    def top_losses(self, k: int | None = None, *, largest: bool = True):
        if self.losses is None:
            raise ValueError("per-sample losses were not provided")
        if self.losses.ndim != 1:
            raise ValueError("losses must be one-dimensional")
        count = len(self.losses) if k is None else k
        if count < 0:
            raise ValueError("k must be non-negative")
        count = min(count, len(self.losses))
        return torch.topk(self.losses, count, largest=largest)


@dataclass
class ClassificationInterpretation(Interpretation):
    class_names: Optional[list[str]] = None

    @property
    def predicted_classes(self):
        if self.predictions.ndim != 2:
            raise ValueError("predictions must have shape [N, C]")
        return self.predictions.argmax(dim=1)

    def confusion_matrix(self, num_classes: int | None = None) -> torch.Tensor:
        if self.targets.ndim != 1:
            raise ValueError("targets must be class indices with shape [N]")
        if num_classes is None:
            num_classes = (self.predictions.shape[1] if self.predictions.ndim == 2
                           else int(self.targets.max().item()) + 1 if self.targets.numel() else 0)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive for an empty or classless dataset")
        truth = self.targets.to(torch.long)
        pred = self.predicted_classes.to(torch.long)
        if truth.numel() and (truth.min() < 0 or truth.max() >= num_classes):
            raise ValueError("target class index outside [0, num_classes)")
        encoded = truth * num_classes + pred
        return torch.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    def most_confused(self, min_count: int = 1):
        if min_count < 1:
            raise ValueError("min_count must be >= 1")
        cm = self.confusion_matrix().clone()
        cm.fill_diagonal_(0)
        pairs = []
        for actual, predicted in torch.nonzero(cm >= min_count, as_tuple=False).tolist():
            count = int(cm[actual, predicted])
            actual_name = self.class_names[actual] if self.class_names else str(actual)
            predicted_name = self.class_names[predicted] if self.class_names else str(predicted)
            pairs.append((actual_name, predicted_name, count))
        return sorted(pairs, key=lambda item: item[2], reverse=True)

    def plot_confusion_matrix(self, *, normalize: bool = False, ax=None, title="Confusion matrix"):
        """Plot using matplotlib (optional dependency); returns the Axes."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("install matplotlib to plot confusion matrices") from exc
        cm = self.confusion_matrix().cpu().numpy().astype(float)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = cm / row_sums.clip(min=1)
        if ax is None:
            _, ax = plt.subplots()
        image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(image, ax=ax)
        names = self.class_names or [str(i) for i in range(cm.shape[0])]
        ax.set(xticks=range(len(names)), yticks=range(len(names)),
               xticklabels=names, yticklabels=names,
               xlabel="Predicted", ylabel="Actual", title=title)
        ax.tick_params(axis="x", labelrotation=90)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                        ha="center", va="center")
        ax.figure.tight_layout()
        return ax


@dataclass
class MultiLabelClassificationInterpretation(Interpretation):
    threshold: float = 0.3
    from_logits: bool = True
    class_names: Optional[list[str]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.predictions.ndim != 2 or self.targets.shape != self.predictions.shape:
            raise ValueError("multi-label predictions and targets must both have shape [N, C]")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

    @property
    def probabilities(self):
        return self.predictions.sigmoid() if self.from_logits else self.predictions

    @property
    def predicted_labels(self):
        return self.probabilities >= self.threshold

    def per_class_confusion(self):
        pred = self.predicted_labels
        truth = self.targets.bool()
        tp = (pred & truth).sum(dim=0)
        fp = (pred & ~truth).sum(dim=0)
        fn = (~pred & truth).sum(dim=0)
        tn = (~pred & ~truth).sum(dim=0)
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
'''

files["training_extensions/engine.py"] = r'''
"""A compact reference training loop with AMP, accumulation and validation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional
import torch
from torch import nn
from .callbacks import GradientAccumulator, clip_gradients, freeze_batchnorm_stats
from .precision import PrecisionConfig, autocast_context, make_grad_scaler


@dataclass
class EpochResult:
    train_loss: float
    validation_loss: Optional[float]
    optimizer_steps: int


@dataclass
class FitResult:
    epochs: list[EpochResult] = field(default_factory=list)


def _move_batch(batch, device):
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("each batch must be a tuple/list (inputs, targets)")
    return batch[0].to(device), batch[1].to(device)


def fit(
    model: nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    *,
    epochs: int,
    device: torch.device | str,
    validation_loader: Optional[Iterable] = None,
    scheduler=None,
    precision: PrecisionConfig = PrecisionConfig(),
    accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    freeze_bn: bool = False,
) -> FitResult:
    """Train a model. Scheduler is stepped after each successful optimizer step.

    Loss functions are expected to return a mean-reduced scalar for each batch.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    device = torch.device(device)
    model.to(device)
    scaler = make_grad_scaler(device, precision)
    accumulator = GradientAccumulator(accumulation_steps)
    history = FitResult()

    for _ in range(epochs):
        model.train()
        if freeze_bn:
            freeze_batchnorm_stats(model)
        total_loss, total_samples, steps = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for batch in train_loader:
            inputs, targets = _move_batch(batch, device)
            batch_size = inputs.shape[0]
            with autocast_context(device, precision):
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar mean-reduced loss")
            total_loss += float(loss.detach()) * batch_size
            total_samples += batch_size

            if scaler is None:
                accumulator.backward(loss, batch_size)
            else:
                scaler.scale(loss * batch_size).backward()
                accumulator.micro_batches += 1
                accumulator.samples += batch_size

            if accumulator.ready():
                if max_grad_norm is not None and scaler is not None:
                    scaler.unscale_(optimizer)
                if max_grad_norm is not None:
                    clip_gradients(model, max_grad_norm)
                did_step = accumulator.step(optimizer, model, scaler=scaler)
                if did_step:
                    steps += 1
                    if scheduler is not None:
                        scheduler.step()

        if accumulator.micro_batches:
            if accumulator.drop_last:
                optimizer.zero_grad(set_to_none=True)
                accumulator.reset()
            else:
                if max_grad_norm is not None and scaler is not None:
                    scaler.unscale_(optimizer)
                if max_grad_norm is not None:
                    clip_gradients(model, max_grad_norm)
                if accumulator.step(optimizer, model, scaler=scaler):
                    steps += 1
                    if scheduler is not None:
                        scheduler.step()

        val_loss = None
        if validation_loader is not None:
            model.eval()
            val_total, val_samples = 0.0, 0
            with torch.inference_mode():
                for batch in validation_loader:
                    inputs, targets = _move_batch(batch, device)
                    with autocast_context(device, precision):
                        loss = loss_fn(model(inputs), targets)
                    val_total += float(loss) * inputs.shape[0]
                    val_samples += inputs.shape[0]
            val_loss = val_total / val_samples if val_samples else float("nan")

        history.epochs.append(EpochResult(
            total_loss / total_samples if total_samples else float("nan"),
            val_loss, steps,
        ))
    return history
'''

files["tests/test_core.py"] = r'''
import unittest
import torch
from torch import nn
from torch.optim import SGD

from training_extensions import (
    mixup_batch, GradientAccumulator, clip_gradients,
    ClassificationInterpretation, MultiLabelClassificationInterpretation,
    build_one_cycle_scheduler, build_flat_cosine_scheduler, find_learning_rate,
)


class CoreTests(unittest.TestCase):
    def test_mixup(self):
        x = torch.randn(4, 3)
        y = torch.tensor([0, 1, 0, 1])
        result = mixup_batch(x, y, alpha=0.4)
        self.assertEqual(result.inputs.shape, x.shape)
        self.assertEqual(result.targets_a.shape, y.shape)
        self.assertGreaterEqual(result.lam, 0)
        self.assertLessEqual(result.lam, 1)

    def test_gradient_accumulation_partial_window(self):
        model = nn.Linear(1, 1, bias=False)
        opt = SGD(model.parameters(), lr=0.1)
        acc = GradientAccumulator(2)
        opt.zero_grad(set_to_none=True)
        for _ in range(3):
            loss = model(torch.ones(1, 1)).sum()
            acc.backward(loss, 1)
            if acc.ready():
                acc.step(opt, model)
        self.assertEqual(acc.micro_batches, 1)
        self.assertTrue(acc.step(opt, model))
        self.assertEqual(acc.micro_batches, 0)

    def test_confusion_matrix(self):
        pred = torch.tensor([[.9, .1], [.2, .8], [.7, .3]])
        target = torch.tensor([0, 0, 1])
        obj = ClassificationInterpretation(pred, target)
        self.assertEqual(obj.confusion_matrix().tolist(), [[1, 1], [1, 0]])

    def test_multilabel(self):
        logits = torch.tensor([[4., -4.], [-4., 4.]])
        targets = torch.tensor([[1, 0], [0, 1]])
        obj = MultiLabelClassificationInterpretation(logits, targets)
        self.assertEqual(obj.predicted_labels.tolist(), [[True, False], [False, True]])
        counts = obj.per_class_confusion()
        self.assertEqual(counts["tp"].tolist(), [1, 1])

    def test_schedulers(self):
        model = nn.Linear(2, 2)
        opt = SGD(model.parameters(), lr=0.1)
        one = build_one_cycle_scheduler(opt, max_lr=0.1, total_steps=5)
        for _ in range(5):
            opt.step()
            one.step()
        flat = build_flat_cosine_scheduler(opt, total_steps=5)
        for _ in range(5):
            opt.step()
            flat.step()

    def test_lr_finder_restores_model(self):
        model = nn.Linear(2, 2)
        opt = SGD(model.parameters(), lr=0.01)
        data = [(torch.randn(4, 2), torch.randint(0, 2, (4,))) for _ in range(3)]
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = find_learning_rate(
            model, opt, lambda out, y: nn.functional.cross_entropy(out, y),
            data, device="cpu", start_lr=1e-4, end_lr=0.1, num_steps=5,
        )
        self.assertGreaterEqual(len(result.losses), 2)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))

    def test_gradient_clipping_validation(self):
        model = nn.Linear(2, 1)
        with self.assertRaises(ValueError):
            clip_gradients(model, 0)


if __name__ == "__main__":
    unittest.main()
'''

files["pyproject.toml"] = r'''
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "modern-training-extensions"
version = "1.0.0"
description = "Modern PyTorch training utilities migrated from legacy fastai v1 extensions"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["torch>=2.1"]

[project.optional-dependencies]
plot = ["matplotlib>=3.7"]
test = ["pytest>=7.4"]

[tool.setuptools.packages.find]
include = ["training_extensions*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
'''

files["README.md"] = r'''
# Modern Training Extensions

A modern PyTorch replacement for the uploaded fastai v1 training-extension module.
The package preserves the useful ideas—learning-rate scheduling and finding,
mixed precision, MixUp, gradient clipping, BatchNorm-stat freezing, gradient
accumulation, and classification interpretation—without patching framework
classes or depending on fastai internals.

## Compatibility

- Python: 3.10+
- PyTorch: 2.1+
- Optional plotting: Matplotlib 3.7+
- No fastai dependency.
- This is an API redesign, not a drop-in fastai v1 compatibility shim.
- The original `Learner` monkey-patches (`fit_one_cycle`, `to_fp16`, etc.)
  are intentionally replaced by explicit functions and standard PyTorch
  optimizers/schedulers.
- `OneCycleLR` and the flat-cosine scheduler are stepped once per optimizer
  update. With gradient accumulation, count optimizer updates, not micro-batches.

## Install

```bash
python -m pip install -e ".[test,plot]"
```

## Quick start

```python
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from training_extensions import (
    fit, PrecisionConfig, build_one_cycle_scheduler,
)

x = torch.randn(1024, 20)
y = torch.randint(0, 4, (1024,))
loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4))
optimizer = AdamW(model.parameters(), lr=1e-3)

# 32 batches/epoch; with no accumulation, 5 epochs = 160 optimizer updates.
scheduler = build_one_cycle_scheduler(
    optimizer, max_lr=1e-3, epochs=5, steps_per_epoch=len(loader)
)

result = fit(
    model, loader, optimizer, nn.CrossEntropyLoss(),
    epochs=5, device="cuda" if torch.cuda.is_available() else "cpu",
    scheduler=scheduler,
    precision=PrecisionConfig("fp16" if torch.cuda.is_available() else "fp32"),
    accumulation_steps=1,
)
print(result.epochs[-1])
```

## Feature mapping

| Uploaded fastai v1 feature | Modern replacement |
|---|---|
| `fit_linear_cycle` / `fit_one_cycle` | `build_one_cycle_scheduler` + `fit` |
| `fit_fc` | `build_flat_cosine_scheduler` + `fit` |
| `lr_find` | `find_learning_rate` |
| `to_fp16` / `to_fp32` | `PrecisionConfig`, `autocast_context`, `make_grad_scaler` |
| `mixup` | `mixup_batch` |
| `GradientClipping` / `clip_grad` | `clip_gradients` or `fit(max_grad_norm=...)` |
| `BnFreeze` | `freeze_batchnorm_stats` |
| `AccumulateScheduler` | `GradientAccumulator` or `fit(accumulation_steps=...)` |
| `ClassificationInterpretation` | `ClassificationInterpretation` |
| unfinished multilabel interpretation | `MultiLabelClassificationInterpretation` |
| `ShowGraph` | use returned `FitResult` with your plotting stack |

## MixUp

```python
from training_extensions import mixup_batch

mixed = mixup_batch(inputs, targets, alpha=0.4)
logits = model(mixed.inputs)
loss = (
    mixed.lam * loss_fn(logits, mixed.targets_a)
    + (1 - mixed.lam) * loss_fn(logits, mixed.targets_b)
)
```

## Learning-rate finder

```python
from training_extensions import find_learning_rate

result = find_learning_rate(
    model, optimizer, loss_fn, train_loader,
    device="cuda", start_lr=1e-7, end_lr=1.0, num_steps=100,
)
print(result.suggested_lr)
```

The finder restores model weights, optimizer state, and the model's original
train/eval mode. It requires a re-iterable batch source such as a DataLoader.
For stochastic models, RNG state and external data-loader side effects are not
rewound.

## Interpretation

```python
from training_extensions import ClassificationInterpretation

analysis = ClassificationInterpretation(
    predictions=logits.detach(),
    targets=labels,
    losses=per_sample_losses,
    class_names=["cat", "dog", "bird"],
)
cm = analysis.confusion_matrix()
errors = analysis.most_confused(min_count=2)
```

For multilabel classification, pass a `[N, C]` tensor of logits and a matching
binary target tensor. Set `from_logits=False` if predictions are already
probabilities.

## Full tests

```bash
python -m pip install -e ".[test]"
python -m pytest
```

## Important engineering notes

1. The training loop assumes a scalar mean-reduced loss. It weights each
   micro-batch loss by its sample count before backward, then normalizes the
   accumulated gradient by the total samples in the update window.
2. If using a custom loss with `reduction="sum"`, adapt the loss weighting;
   do not pass it unchanged to `fit`.
3. For CUDA FP16, gradients are scaled and unscaled before clipping. BF16 and
   FP32 do not use GradScaler in this implementation.
4. Call `freeze_batchnorm_stats` after `model.train()`; the included `fit`
   helper does this at each epoch start when `freeze_bn=True`.
5. The learning-rate finder is an experiment: although model and optimizer
   state are restored, data-loader random state and arbitrary user callback
   side effects are not.
6. Distributed training, gradient checkpointing, EMA, optimizer sharding,
   and model-specific multi-input batch structures are intentionally outside
   this first implementation. Extend the reference loop for those requirements.
'''

for rel, content in files.items():
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")

# Add a simple license placeholder? avoid inventing license. Add source attribution note.
(root / "SOURCE_NOTES.md").write_text(
    "# Source and migration notes\n\n"
    "This implementation was derived from the user-provided fastai v1 source excerpt "
    "uploaded in this conversation. It preserves the feature intent while replacing "
    "fastai-specific callbacks and monkey-patching with explicit modern PyTorch APIs. "
    "No external repository or source was linked.\n",
    encoding="utf-8"
)

zip_path = Path("/mnt/data/modern_training_extensions.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in root.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(root.parent))

print(f"Created {len(files)+1} files in {root}")
print(f"Archive: {zip_path}")
