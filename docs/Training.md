# Native LMLM training adapter

The supported training adapter lives at `model_training/lmlm_training`. The historical
`model_training.Lmlm_training` import remains available through a compatibility module.

## Responsibility boundary
Fastai1 supplies generic PyTorch training utilities. LMLM remains the owner of its
architecture, tokenizer, dataset interpretation, model artifacts, and runtime. Pass an
already-constructed `torch.nn.Module`; this package does not fetch model weights or
construct an LMLM architecture.

The adapter is independent of inference/processor protocols. Optimizers, gradient
scaling, and training checkpoints are training concerns and should not be added to an
inference transport contract.

## Basic usage
    from model_training.lmlm_training import TrainConfig, train

    result = train(model, train_loader, optimizer, loss_fn,
        config=TrainConfig(epochs=3, accumulation_steps=2, device="auto",
                           precision="auto", max_grad_norm=1.0),
        validation_loader=validation_loader)

Batches may be `(inputs, targets)` or dictionaries containing `inputs` and `targets`.
The model is called as `model(inputs)`; custom multi-argument model signatures should
be adapted by the caller before passing the batch.

## Automatic mixed precision (AMP)

The supported pure-Python AMP API is `model_training.lmlm_training.amp`:

```python
import torch
from model_training.lmlm_training import autocast

with autocast(device_type="cuda", dtype=torch.float16):
    output = model(inputs)

with autocast(device_type="cpu", dtype=torch.bfloat16):
    output = model(inputs)
```

`autocast()` defaults to CUDA FP16 for compatibility with the historical CUDA AMP
surface. Specify `device_type` explicitly for CPU or other supported PyTorch devices.
`custom_fwd` and `custom_bwd` preserve autocast state for custom autograd functions;
`custom_fwd(cast_inputs=...)` casts floating tensors on the selected device and disables
autocast while forward runs. FP64 and non-floating tensors are not cast. NumPy arrays,
strings, and bytes are treated as atomic values by the recursive caster.

The legacy `TORCH.CUDA.AMP/autocast.pyx` is retained as historical Cython source. It is
not the supported Python import path and is not compiled by the current package build.
The maintained implementation is `model_training.lmlm_training.amp`.

## Device and precision
- `auto` prefers CUDA, then Apple MPS, then CPU.
- Automatic precision selects BF16 or FP16 on supported CUDA devices and FP32 on CPU/MPS.
- FP16 scaling is enabled only for CUDA.
- CPU BF16 is explicit opt-in and depends on installed PyTorch operator support.
- Hardware-specific CUDA/MPS behavior must be tested on the target device. CPU tests do
  not establish GPU correctness.

## Checkpoints and trust
`CheckpointManager.save` writes to a temporary file and atomically replaces the destination.
`load_checkpoint` restores model and any supplied optimizer, scheduler, or scaler state.
PyTorch serialization can execute code during deserialization: load only trusted files.
Do not load arbitrary user-uploaded checkpoints.

A checkpoint records epoch, global step, best metric, metadata, and RNG state. The caller
controls resume orchestration and should construct the model, optimizer, scheduler, and
scaler consistently before restoring.

## Benchmarks
`benchmark` reports measured latency percentiles, throughput, and CUDA peak allocated
memory when available. Results depend on hardware, input shape, warmup, and runtime
conditions; they are measurements, not performance guarantees.

## Legacy code and Cython
Top-level fastai-v1-derived modules and `.pyx` sources are retained. They are not yet
guaranteed to form an installable, fully tested legacy package. In particular, historical
`Core.py` references `imports.core`; that dependency surface must be restored or migrated
before claiming the old top-level API is importable. Cython extensions are not built
automatically by the Python package configuration.

## Test
    python -m pip install -e ".[training,dev]"
    pytest
