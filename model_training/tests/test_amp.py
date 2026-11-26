from collections import namedtuple

import pytest
import torch

from model_training.lmlm_training.amp import (
    _autocast_dtype, _autocast_enabled, _cast, autocast, custom_bwd, custom_fwd,
)


def test_autocast_disabled_context_is_noop_on_cpu():
    with autocast(enabled=False, device_type="cpu"):
        assert not _autocast_enabled("cpu")


def test_cpu_autocast_context_uses_requested_dtype():
    with autocast(device_type="cpu", dtype=torch.bfloat16):
        assert _autocast_enabled("cpu")
        assert _autocast_dtype("cpu") == torch.bfloat16


def test_cast_only_changes_floating_tensors_on_selected_device():
    fp32 = torch.ones(2, dtype=torch.float32)
    fp64 = torch.ones(2, dtype=torch.float64)
    integer = torch.ones(2, dtype=torch.int64)
    nested = {"x": [fp32, fp64, integer], "label": "unchanged"}
    casted = _cast(nested, torch.bfloat16, device_type="cpu")
    assert casted["x"][0].dtype == torch.bfloat16
    assert casted["x"][1].dtype == torch.float64
    assert casted["x"][2].dtype == torch.int64
    assert casted["label"] == "unchanged"


def test_cast_preserves_namedtuple_shape():
    Pair = namedtuple("Pair", "left right")
    pair = Pair(torch.ones(1), "text")
    result = _cast(pair, torch.float16, device_type="cpu")
    assert isinstance(result, Pair)
    assert result.left.dtype == torch.float16
    assert result.right == "text"


def test_custom_autograd_forward_backward_preserves_cpu_autocast():
    class Scale(torch.autograd.Function):
        @staticmethod
        @custom_fwd(device_type="cpu")
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * 2

        @staticmethod
        @custom_bwd
        def backward(ctx, grad):
            return grad * 2

    x = torch.ones(3, requires_grad=True)
    with autocast(device_type="cpu", dtype=torch.bfloat16):
        Scale.apply(x).sum().backward()
    assert torch.equal(x.grad, torch.full_like(x, 2))


def test_custom_fwd_casts_nested_inputs_and_disables_autocast():
    class Inspect(torch.autograd.Function):
        @staticmethod
        @custom_fwd(cast_inputs=torch.bfloat16, device_type="cpu")
        def forward(ctx, x):
            ctx.seen_dtype = x.dtype
            ctx.seen_autocast = _autocast_enabled("cpu")
            return x.clone()

        @staticmethod
        @custom_bwd
        def backward(ctx, grad):
            return grad

    x = torch.ones(2, requires_grad=True)
    with autocast(device_type="cpu", dtype=torch.bfloat16):
        y = Inspect.apply(x)
    assert y.dtype == torch.bfloat16
    assert not _autocast_enabled("cpu")
    y.float().sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_autocast_decorator():
    @autocast(device_type="cpu", enabled=False)
    def identity(x):
        return x
    x = torch.ones(1)
    assert torch.equal(identity(x), x)
