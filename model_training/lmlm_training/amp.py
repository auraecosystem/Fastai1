"""CUDA automatic mixed precision helpers for custom autograd functions.

Preserves historical torch.cuda.amp behavior while using device-aware autocast
queries where available. This module deliberately targets CUDA only.
"""
from __future__ import annotations

import collections.abc
import functools
from typing import Any

import torch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

__all__ = ["autocast", "custom_fwd", "custom_bwd"]


class autocast(torch.autocast):
    """CUDA autocast context manager and decorator."""

    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.float16,
                 cache_enabled: bool = True) -> None:
        super().__init__("cuda", enabled=enabled, dtype=dtype,
                         cache_enabled=cache_enabled)


def _is_autocast_enabled() -> bool:
    """Read CUDA autocast state across PyTorch API generations."""
    try:
        return bool(torch.is_autocast_enabled("cuda"))
    except TypeError:
        return bool(torch.is_autocast_enabled())


def _get_autocast_dtype() -> torch.dtype:
    """Read the active CUDA autocast dtype with a legacy fallback."""
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is not None:
        try:
            return getter("cuda")
        except (TypeError, RuntimeError):
            pass
    legacy_getter = getattr(torch, "get_autocast_gpu_dtype", None)
    return legacy_getter() if legacy_getter is not None else torch.float16


def _rebuild_mapping(value, items):
    if isinstance(value, collections.defaultdict):
        return type(value)(value.default_factory, items)
    try:
        return type(value)(items)
    except (TypeError, ValueError):
        return dict(items)


def _cast(value: Any, dtype: torch.dtype) -> Any:
    """Recursively cast eligible CUDA floating tensors; preserve common containers."""
    if isinstance(value, torch.Tensor):
        eligible = (value.is_floating_point() and value.is_cuda
                    and value.dtype != torch.float64)
        return value.to(dtype=dtype) if eligible else value
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        return value
    if np is not None and isinstance(value, np.ndarray):
        return value
    if isinstance(value, collections.abc.Mapping):
        items = [(_cast(k, dtype), _cast(v, dtype)) for k, v in value.items()]
        return _rebuild_mapping(value, items)
    if isinstance(value, list):
        converted = [_cast(item, dtype) for item in value]
        try:
            return type(value)(converted)
        except TypeError:
            return converted
    if isinstance(value, tuple):
        converted = tuple(_cast(item, dtype) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*converted)
        try:
            return type(value)(converted)
        except TypeError:
            return converted
    if isinstance(value, set):
        return type(value)(_cast(item, dtype) for item in value)
    if isinstance(value, frozenset):
        return type(value)(_cast(item, dtype) for item in value)
    if isinstance(value, collections.abc.Iterable):
        return (_cast(item, dtype) for item in value)
    return value


def custom_fwd(fwd=None, *, cast_inputs: torch.dtype | None = None):
    """Decorate a custom autograd forward method.

    When cast_inputs is set and CUDA autocast is active, eligible floating CUDA
    tensors are cast and the forward body runs with autocast disabled. Outside
    autocast, casting is a no-op.
    """
    if fwd is None:
        return functools.partial(custom_fwd, cast_inputs=cast_inputs)

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if not args:
            raise TypeError("custom_fwd must decorate an autograd forward method")
        ctx = args[0]
        enabled = _is_autocast_enabled()
        ctx._dtype = _get_autocast_dtype()
        ctx._fwd_used_autocast = enabled
        if cast_inputs is None or not enabled:
            return fwd(*args, **kwargs)
        ctx._fwd_used_autocast = False
        with autocast(enabled=False):
            return fwd(*_cast(args, cast_inputs), **_cast(kwargs, cast_inputs))
    return decorate_fwd


def custom_bwd(bwd):
    """Restore the CUDA autocast state captured by the custom forward method."""
    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        if not args:
            raise TypeError("custom_bwd must decorate an autograd backward method")
        ctx = args[0]
        enabled = bool(getattr(ctx, "_fwd_used_autocast", False))
        dtype = getattr(ctx, "_dtype", torch.float16)
        with autocast(enabled=enabled, dtype=dtype):
            return bwd(*args, **kwargs)
    return decorate_bwd
