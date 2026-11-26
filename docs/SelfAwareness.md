# PyTorch self-awareness / runtime introspection

Fastai1 provides `model_training.self_awareness` as a lightweight observer for
measurable state from a supplied `torch.nn.Module`. It reports architecture and
parameter metadata, gradient health, tensor numerical health, and training
events explicitly recorded by the caller.

This is operational introspection, not consciousness, subjective experience,
autonomous self-knowledge, or unrestricted access to model internals.

## Quick start

```python
import torch
from model_training.self_awareness import SelfAwareness

model = torch.nn.Linear(16, 4)
observer = SelfAwareness(model, history_size=50)

for epoch in range(3):
    observer.start_epoch()
    # In the real loop, pass the detached loss after the optimizer step.
    observer.record_step(torch.tensor(0.42), batch_size=32)

report = observer.inspect()
```

## API

- `ModelIntrospector.inspect(model)`: parameter counts, module types, mode,
  devices, dtypes, buffers, and dense/sparse gradient health.
- `ModelIntrospector.inspect_tensor(tensor)`: shape, dtype, device, finite
  fraction, and descriptive statistics. It uses a detached CPU diagnostic copy,
  avoiding unsupported float64 operations on MPS.
- `TrainingAwareness`: tracks epoch, steps, samples, recent finite losses,
  best loss, non-finite loss count, elapsed time, and caller-supplied metrics.
- `SelfAwareness`: combines model inspection with explicit training observations.

Call `record_step` after a completed optimizer step. The observer installs no
hooks, modifies no model state, retains no batches, and does not change gradients.
Non-finite losses/metrics are represented as JSON null with a separate finite
flag/counter, so reports can be serialized with strict JSON settings.

## Interpretation and limits

Parameter count does not establish capability; task-specific benchmarks are
required. A raw softmax score is not automatically calibrated confidence. This
module does not infer confidence or claim that a model knows its own limitations.
External hardware availability and operating-system metrics are outside scope.
