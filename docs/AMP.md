# CUDA automatic mixed precision helpers

Fastai1 provides a CUDA AMP compatibility module at
model_training.lmlm_training.amp. It preserves the behavior of the supplied
legacy CUDA helper while using device-aware PyTorch autocast state and dtype
queries when available.

## Usage

    import torch
    from model_training.lmlm_training.amp import autocast, custom_fwd, custom_bwd

    class ScaledOperation(torch.autograd.Function):
        @staticmethod
        @custom_fwd(cast_inputs=torch.float16)
        def forward(ctx, value):
            return value * 2

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_output):
            return grad_output * 2

    value = torch.randn(32, 32, device="cuda")
    with autocast(dtype=torch.float16):
        output = ScaledOperation.apply(value)

## Compatibility and scope

- The module targets CUDA autocast and does not enable CPU or MPS autocast.
- autocast is both a context manager and a decorator.
- custom_fwd supports bare and configured decorator forms.
- custom_bwd restores the CUDA autocast-enabled state and dtype captured by forward.
- Recursive casting handles common containers and passes through strings, bytes,
  NumPy arrays, non-floating tensors, CPU tensors, and float64 tensors.
- CUDA execution and dtype behavior require validation on target hardware.
  The module intentionally avoids PyTorch private TorchScript internals.

The module is independently importable within the Fastai1 distribution and is
re-exported from model_training.lmlm_training for training integration.
