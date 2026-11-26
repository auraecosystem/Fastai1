# Native LMLM training adapter

The supported training adapter lives at 'model_training/lmlm_training'. The historical implementation directory 'model_training/Lmlm_training' remains available, and the lowercase package re-exports its public API for predictable imports on case-sensitive filesystems.

## Responsibility boundary
Fastai1 supplies generic PyTorch training utilities. LMLM remains the owner of its architecture, tokenizer, dataset interpretation, model artifacts, and runtime. Pass an already-constructed 'torch.nn.Module'; this package does not fetch model weights or construct an LMLM architecture.

The adapter is independent of inference/processor protocols. Optimizers, gradient scaling, and training checkpoints are training concerns and should not be added to an inference transport contract.

## Basic usage
```python
from model_training.lmlm_training import TrainConfig, train

result = train(model, train_loader, optimizer, loss_fn,
    config=TrainConfig(epochs=3, accumulation_steps=2, device="auto",
                       precision="auto", max_grad_norm=1.0),
    validation_loader=validation_loader)
```

Batches may be '(inputs, targets)' or dictionaries containing 'inputs' and 'targets'. The model is called as 'model(inputs)'; custom multi-argument model signatures should be adapted by the caller before passing the batch.

## Device and precision
- 'auto' prefers CUDA, then Apple MPS, then CPU.
- Automatic precision selects BF16 or FP16 on supported CUDA devices and FP32 on CPU/MPS.
- FP16 scaling is enabled only for CUDA.
- CPU BF16 is explicit opt-in and depends on the installed PyTorch build and operator support; unsupported operations may still fail at runtime.
- Hardware-specific CUDA/MPS behavior must be tested on the target device. CPU tests do not establish GPU correctness.

## Checkpoints and trust
'CheckpointManager.save' writes to a temporary file and atomically replaces the destination. 'load_checkpoint' restores model and any supplied optimizer, scheduler, or scaler state. Checkpoints use PyTorch serialization and can execute code during deserialization: load only trusted files that you created or obtained through a trusted channel. Do not load arbitrary user-uploaded checkpoints.

A checkpoint records epoch, global step, best metric, metadata, and RNG state. The caller controls resume orchestration and should construct the model, optimizer, scheduler, and scaler consistently before restoring.

## Benchmarks
'benchmark' reports measured latency percentiles, throughput, and CUDA peak allocated memory when available. Results depend on hardware, input shape, warmup, and runtime conditions; they are measurements, not performance guarantees.

## Legacy code and Cython
Top-level fastai-v1-derived modules and .pyx sources are retained. They are not yet guaranteed to form an installable, fully tested legacy package. In particular, historical 'Core.py' references 'imports.core'; that dependency surface must be restored or migrated before claiming the old top-level API is importable. Cython extensions are not built automatically by the Python package configuration.

## Test
```bash
python -m pip install -e ".[training,dev]"
pytest
```
