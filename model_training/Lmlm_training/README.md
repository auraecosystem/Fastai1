# LMLM Native Training

A native PyTorch training subsystem for model implementations owned by LMLM. It is
intentionally model-agnostic: pass an existing `torch.nn.Module`, optimizer, data
loader, and loss function. This package does not define or replace the LMLM neural
architecture, tokenizer, model weights, or inference engine.

## Scope

- Hardware selection: CUDA, Apple MPS, or CPU.
- Precision policy: automatic FP32/FP16/BF16 selection with conservative defaults.
- Training and evaluation with sample-weighted gradient accumulation.
- Gradient clipping and optimizer-step scheduler semantics.
- Atomic, resumable checkpoints (model, optimizer, scheduler, scaler, counters, RNG).
- Inference benchmark with warmup, latency percentiles, throughput and CUDA peak memory.

## Quick start

Run from the repository root after installing PyTorch:

```python
import torch
from model_training.lmlm_training import TrainConfig, CheckpointManager, train

model = YourLmlmModel(...)  # construct the actual LMLM model in its owning package
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
result = train(
    model, train_loader, optimizer, loss_fn,
    config=TrainConfig(epochs=3, device="auto", precision="auto",
                       accumulation_steps=4, max_grad_norm=1.0),
    validation_loader=validation_loader,
    checkpoint_manager=CheckpointManager("artifacts/checkpoints"),
)
print(result.history)
```

`YourLmlmModel`, `train_loader`, and `loss_fn` are deliberately placeholders for
the real LMLM implementation and its approved dataset pipeline; this repository's
extension layer must not invent those interfaces.

Resume:

```python
from model_training.lmlm_training import load_checkpoint
state = load_checkpoint("artifacts/checkpoints/last.pt", model=model,
                        optimizer=optimizer, map_location="cpu")
```

Benchmark:

```python
from model_training.lmlm_training import benchmark
print(benchmark(model, sample_batch, warmup=10, iterations=50).to_dict())
```

## Precision behavior

- `auto`: BF16 on CUDA when supported; otherwise CUDA FP16; FP32 on MPS/CPU.
- CUDA FP16 uses `torch.amp.GradScaler`.
- BF16 does not use a gradient scaler.
- CPU BF16 may be explicitly requested; kernel support depends on the installed
  PyTorch build and CPU. MPS defaults to FP32.
- No automatic CPU fallback occurs after an explicitly requested device fails.

## Checkpoint security

PyTorch checkpoints are pickle-based. Load only checkpoints created by a trusted
local training run. Do not load arbitrary checkpoint files from untrusted sources.

## Integration boundary

The LMLMxtend Rust processor bridge is an inference boundary. Training remains in
the LMLM-owned Python/PyTorch model package and should export approved artifacts
through the LMLM model manifest. Do not put optimizer state or training control
messages on the `lmlm.processor.v1` inference protocol.
