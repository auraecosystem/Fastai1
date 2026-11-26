"""Atomic, resumable training checkpoints."""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = 1


def _rng_state():
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    if not state:
        return
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
    if "python" in state:
        random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class CheckpointManager:
    """Write checkpoints atomically so interrupted saves preserve the last good file."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, *, model, optimizer=None, scheduler=None, scaler=None,
             epoch: int, global_step: int, best_metric: float | None = None,
             metadata: dict[str, Any] | None = None) -> Path:
        if not name or Path(name).name != name:
            raise ValueError("checkpoint name must be a plain filename")
        target = self.directory / name
        payload = {
            "format_version": FORMAT_VERSION,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": int(epoch), "global_step": int(global_step),
            "best_metric": best_metric, "metadata": metadata or {},
            "rng_state": _rng_state(),
        }
        fd, temp_name = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=self.directory)
        os.close(fd)
        try:
            torch.save(payload, temp_name)
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return target


def load_checkpoint(path: str | Path, *, model, optimizer=None, scheduler=None,
                    scaler=None, map_location="cpu", restore_rng: bool = True,
                    strict: bool = True) -> dict[str, Any]:
    """Restore a trusted checkpoint created by this package.

    PyTorch serialization can execute code during deserialization. Never load
    checkpoints from untrusted sources.
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported or invalid checkpoint format")
    model.load_state_dict(payload["model"], strict=strict)
    for key, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None and payload.get(key) is not None:
            obj.load_state_dict(payload[key])
    if restore_rng:
        _restore_rng(payload.get("rng_state"))
    return {key: payload.get(key) for key in ("epoch", "global_step", "best_metric", "metadata")}
