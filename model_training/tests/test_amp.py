"""CPU-safe regression tests for CUDA AMP compatibility helpers."""
from collections import defaultdict, namedtuple

import pytest
import torch

from model_training.lmlm_training.amp import _cast, autocast, custom_bwd, custom_fwd


def test_autocast_disabled_context_is_safe_without_cuda():
    with autocast(enabled=False):
        value = torch.tensor([1.0])
    assert value.item() == 1.0


def test_autocast_can_decorate_function():
    @autocast(enabled=False)
    def identity(value):
        return value
    value = torch.tensor([2.0])
    assert identity(value) is value


def test_cast_preserves_cpu_tensors_and_scalar_containers():
    tensor = torch.tensor([1.0])
    result = _cast({"items": [tensor, "text", b"bytes", 4]}, torch.float16)
    assert result["items"][0] is tensor
    assert result["items"][1:] == ["text", b"bytes", 4]


def test_cast_preserves_namedtuple_and_defaultdict():
    Pair = namedtuple("Pair", "left right")
    pair = Pair("a", "b")
    assert _cast(pair, torch.float16) == pair
    source = defaultdict(list, {"key": ["value"]})
    result = _cast(source, torch.float16)
    assert isinstance(result, defaultdict)
    assert result.default_factory is list
    assert result["key"] == ["value"]


def test_cast_handles_generators_lazily():
    source = (item for item in (1, 2, 3))
    assert list(_cast(source, torch.float16)) == [1, 2, 3]


def test_custom_autograd_forward_backward_without_autocast():
    class Multiply(torch.autograd.Function):
        @staticmethod
        @custom_fwd
        def forward(ctx, value):
            return value * 2

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            return grad_output * 2
    value = torch.tensor([3.0], requires_grad=True)
    Multiply.apply(value).sum().backward()
    assert value.grad.item() == 2.0


def test_custom_fwd_supports_bare_and_configured_forms():
    @custom_fwd
    def bare(ctx, value):
        return value

    @custom_fwd(cast_inputs=torch.float16)
    def configured(ctx, value):
        return value

    ctx = type("Context", (), {})()
    value = torch.tensor([1.0])
    assert bare(ctx, value) is value
    assert configured(ctx, value) is value
    assert ctx._fwd_used_autocast is False


def test_numpy_arrays_are_passthrough_when_installed():
    np = pytest.importorskip("numpy")
    value = np.array([1.0], dtype=np.float32)
    assert _cast(value, torch.float16) is value
