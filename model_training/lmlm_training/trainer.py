"""Generic native PyTorch train/evaluate loops for an LMLM-owned nn.Module."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any
import math
import torch
from torch import nn
from .device import DevicePlan, resolve_device_plan

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1
    accumulation_steps: int = 1
    max_grad_norm: float | None = None
    log_every_steps: int = 0
    device: str = "auto"
    precision: str = "auto"

    def __post_init__(self):
        if self.epochs < 1 or self.accumulation_steps < 1:
            raise ValueError("epochs and accumulation_steps must be positive")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")

@dataclass
class TrainResult:
    history: list[dict[str, float]]
    global_step: int
    device_plan: DevicePlan

def _unpack(batch, device):
    if isinstance(batch, dict):
        if "inputs" not in batch or "targets" not in batch:
            raise ValueError("dict batches must contain inputs and targets")
        x, y = batch["inputs"], batch["targets"]
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise ValueError("batch must be (inputs, targets) or a dict with those keys")
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=(device.type == "cuda"))
        if isinstance(value, (tuple, list)):
            return type(value)(move(v) for v in value)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value
    return move(x), move(y)

def _batch_size(x, y):
    for value in (x, y):
        if isinstance(value, torch.Tensor) and value.ndim:
            return int(value.shape[0])
        if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
            return int(value[0].shape[0])
    raise ValueError("cannot infer batch size from batch tensors")

def evaluate(model: nn.Module, loader: Iterable, loss_fn: Callable, *,
             device: str | torch.device = "cpu", precision: str = "fp32") -> dict[str, float]:
    plan = resolve_device_plan(str(device), precision)
    model.eval()
    total_loss, total_samples, batches = 0.0, 0, 0
    with torch.inference_mode():
        for batch in loader:
            x, y = _unpack(batch, plan.device)
            size = _batch_size(x, y)
            with plan.autocast():
                loss = loss_fn(model(x), y)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("loss_fn must return a finite scalar mean loss")
            total_loss += float(loss.detach()) * size
            total_samples += size
            batches += 1
    return {"loss": total_loss / total_samples if total_samples else math.nan,
            "samples": float(total_samples), "batches": float(batches)}

def train(model: nn.Module, train_loader: Iterable, optimizer: torch.optim.Optimizer,
          loss_fn: Callable, *, config: TrainConfig = TrainConfig(),
          validation_loader: Iterable | None = None, scheduler=None,
          checkpoint_manager=None, checkpoint_name: str = "last.pt",
          start_epoch: int = 0, global_step: int = 0,
          on_step: Callable[[int, float], Any] | None = None,
          self_awareness: Any | None = None) -> TrainResult:
    if self_awareness is not None and not all(
        callable(getattr(self_awareness, name, None))
        for name in ("start_epoch", "record_step")
    ):
        raise TypeError("self_awareness must provide start_epoch() and record_step()")

    plan = resolve_device_plan(config.device, config.precision)
    model.to(plan.device)
    scaler = plan.make_grad_scaler()
    history = []
    for epoch in range(start_epoch, start_epoch + config.epochs):
        model.train()
        if self_awareness is not None:
            self_awareness.start_epoch(epoch + 1)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, samples, pending_samples, pending_batches = 0.0, 0, 0, 0
        pending_loss_sum = 0.0
        epoch_steps = 0
        for batch in train_loader:
            x, y = _unpack(batch, plan.device)
            size = _batch_size(x, y)
            with plan.autocast():
                loss = loss_fn(model(x), y)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("loss_fn must return a finite scalar mean loss")
            (scaler.scale(loss * size) if scaler else loss * size).backward()
            epoch_loss += float(loss.detach()) * size
            samples += size
            pending_samples += size
            pending_batches += 1
            pending_loss_sum += float(loss.detach()) * size
            if pending_batches >= config.accumulation_steps:
                if scaler:
                    scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending_samples)
                if config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                   error_if_nonfinite=True)
                if scaler:
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    did_update = scaler.get_scale() >= old_scale
                else:
                    optimizer.step()
                    did_update = True
                if did_update:
                    global_step += 1
                    epoch_steps += 1
                    if scheduler is not None:
                        scheduler.step()
                    if on_step is not None:
                        on_step(global_step, float(loss.detach()))
                    if self_awareness is not None:
                        self_awareness.record_step(
                            pending_loss_sum / pending_samples,
                            batch_size=pending_samples,
                        )
                optimizer.zero_grad(set_to_none=True)
                pending_samples, pending_batches, pending_loss_sum = 0, 0, 0.0
        if pending_batches:
            if scaler:
                scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending_samples)
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                               error_if_nonfinite=True)
            if scaler:
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_update = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
                did_update = True
            optimizer.zero_grad(set_to_none=True)
            if did_update:
                global_step += 1
                epoch_steps += 1
                if scheduler is not None:
                    scheduler.step()
                if self_awareness is not None:
                    self_awareness.record_step(
                        pending_loss_sum / pending_samples,
                        batch_size=pending_samples,
                    )
            optimizer.zero_grad(set_to_none=True)
        row = {"epoch": float(epoch + 1), "train_loss": epoch_loss / samples if samples else math.nan,
               "optimizer_steps": float(epoch_steps)}
        if validation_loader is not None:
            val = evaluate(model, validation_loader, loss_fn, device=str(plan.device),
                           precision=plan.precision)
            row["validation_loss"] = val["loss"]
        history.append(row)
        if checkpoint_manager is not None:
            checkpoint_manager.save(checkpoint_name, model=model, optimizer=optimizer,
                                    scheduler=scheduler, scaler=scaler, epoch=epoch + 1,
                                    global_step=global_step, metadata={"device": str(plan.device),
                                    "precision": plan.precision})
    return TrainResult(history, global_step, plan)
