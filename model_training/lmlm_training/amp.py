"""Device-aware automatic mixed precision helpers.

The API mirrors the useful subset of torch.cuda.amp while supporting CPU/CUDA
autocast through PyTorch's public torch.autocast API. CUDA is not required at
import time.
"""
from __future__ import annotations

import collections.abc
import functools
from typing import Any, Callable

import torch

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

__all__ = ["autocast", "custom_fwd", "custom_bwd"]


def _autocast_enabled(device_type: str) -> bool:
    """Read autocast state across PyTorch versions."""
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        if device_type == "cuda":
            return bool(torch.is_autocast_enabled())
        getter = getattr(torch, "is_autocast_cpu_enabled", None)
        return bool(getter()) if device_type == "cpu" and getter else False


def _autocast_dtype(device_type: str) -> torch.dtype:
    """Read the active autocast dtype with compatibility fallbacks."""
    getter = getattr(torch, "get_autocast_dtype", None)
    if getter is not None:
        try:
            return getter(device_type)
        except (TypeError, RuntimeError):
            pass
    if device_type == "cuda":
        legacy = getattr(torch, "get_autocast_gpu_dtype", None)
        return legacy() if legacy is not None else torch.float16
    if device_type == "cpu":
        legacy = getattr(torch, "get_autocast_cpu_dtype", None)
        return legacy() if legacy is not None else torch.bfloat16
    raise ValueError(f"autocast dtype lookup is unsupported for {device_type!r}")


class autocast:
    """Context manager/decorator for autocast on a selected device.

    The default remains CUDA FP16 for compatibility. Specify device_type and
    dtype explicitly for CPU or other supported PyTorch devices.
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype | None = None,
        cache_enabled: bool = True,
        *,
        device_type: str = "cuda",
    ) -> None:
        self.device_type = device_type
        self.enabled = bool(enabled)
        self.dtype = dtype or (
            torch.float16 if device_type == "cuda" else torch.bfloat16
        )
        self.cache_enabled = cache_enabled
        self._context = None

    def __enter__(self):
        self._context = torch.autocast(
            device_type=self.device_type,
            enabled=self.enabled,
            dtype=self.dtype,
            cache_enabled=self.cache_enabled,
        )
        return self._context.__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        if self._context is None:
            raise RuntimeError("autocast context was not entered")
        try:
            return self._context.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self._context = None

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def decorate(*args, **kwargs):
            with type(self)(
                enabled=self.enabled,
                dtype=self.dtype,
                cache_enabled=self.cache_enabled,
                device_type=self.device_type,
            ):
                return func(*args, **kwargs)
        return decorate


def _cast(value: Any, dtype: torch.dtype, *, device_type: str = "cuda") -> Any:
    """Cast eligible floating tensors in common nested Python containers."""
    if isinstance(value, torch.Tensor):
        eligible = (
            value.device.type == device_type
            and value.is_floating_point()
            and value.dtype != torch.float64
        )
        return value.to(dtype=dtype) if eligible else value
    if isinstance(value, (str, bytes, bytearray)):
        return value
    if np is not None and isinstance(value, np.ndarray):
        return value
    if isinstance(value, collections.abc.Mapping):
        items = [
            (_cast(k, dtype, device_type=device_type),
             _cast(v, dtype, device_type=device_type))
            for k, v in value.items()
        ]
        try:
            return type(value)(items)
        except (TypeError, ValueError):
            return dict(items)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(
            *(_cast(v, dtype, device_type=device_type) for v in value)
        )
    if isinstance(value, tuple):
        return tuple(_cast(v, dtype, device_type=device_type) for v in value)
    if isinstance(value, list):
        return [_cast(v, dtype, device_type=device_type) for v in value]
    if isinstance(value, set):
        return {_cast(v, dtype, device_type=device_type) for v in value}
    if isinstance(value, frozenset):
        return frozenset(_cast(v, dtype, device_type=device_type) for v in value)
    if isinstance(value, collections.abc.Iterator):
        return (_cast(v, dtype, device_type=device_type) for v in value)
    return value


def custom_fwd(
    fwd=None, *, cast_inputs: torch.dtype | None = None, device_type: str = "cuda"
):
    """Decorate a custom autograd forward method.

    If cast_inputs is set, eligible inputs are cast only while autocast is
    active; the wrapped forward then executes with autocast disabled.
    """
    if fwd is None:
        return functools.partial(
            custom_fwd, cast_inputs=cast_inputs, device_type=device_type
        )

    @functools.wraps(fwd)
    def decorate_fwd(ctx, *args, **kwargs):
        active = _autocast_enabled(device_type)
        ctx._lmlm_amp_device_type = device_type
        ctx._lmlm_amp_dtype = _autocast_dtype(device_type)
        ctx._lmlm_amp_enabled = active
        if cast_inputs is not None and active:
            with autocast(
                enabled=False, device_type=device_type, dtype=ctx._lmlm_amp_dtype
            ):
                cast_args = _cast(args, cast_inputs, device_type=device_type)
                cast_kwargs = _cast(kwargs, cast_inputs, device_type=device_type)
                return fwd(ctx, *cast_args, **cast_kwargs)
        return fwd(ctx, *args, **kwargs)

    return decorate_fwd


def custom_bwd(bwd):
    """Run custom autograd backward under the forward autocast state."""
    @functools.wraps(bwd)
    def decorate_bwd(ctx, *args, **kwargs):
        device_type = getattr(ctx, "_lmlm_amp_device_type", "cuda")
        dtype = getattr(ctx, "_lmlm_amp_dtype", torch.float16)
        enabled = getattr(ctx, "_lmlm_amp_enabled", False)
        with autocast(
            enabled=enabled, dtype=dtype, device_type=device_type
        ):
            return bwd(ctx, *args, **kwargs)
    return decorate_bwd
