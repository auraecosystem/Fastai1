# PyTorch self-awareness / runtime introspection

Fastai1 provides `model_training.self_awareness` as a lightweight, evidence-based
observer for a supplied `torch.nn.Module`. It reports public model metadata,
parameter and gradient health, tensor numerical health, and training events that
the caller explicitly records.

This is operational introspection, not a claim of consciousness, subjective
experience, autonomous self-knowledge, or unrestricted access to model internals.

## Quick start

```python
import torch
from model_training.self_awareness import SelfAwareness

model = torch.nn.Linear(16, 4)
observer = SelfAwareness(model, history_size=50)

for epoch in range(3):
    observer.start_epoch()
    # Run the normal training loop.
    loss = torch.tensor(0.42)  # Replace with the actual detached loss.
    observer.record_step(loss, batch_size=32)

report = observer.inspect()
print(report)
```

## API

- `ModelIntrospector.inspect(model)`: parameter counts, module types, mode,
  devices, dtypes, buffers, and gradient health.
- `ModelIntrospector.inspect_tensor(tensor)`: shape, dtype, device, finite
  fraction, and descriptive statistics. Statistics are computed from detached
  values and do not retain the computation graph.
- `TrainingAwareness`: tracks epoch, steps, samples, recent finite losses,
  best loss, non-finite loss count, elapsed time, and caller-supplied metrics.
- `SelfAwareness`: combines model inspection with explicit training observations.

Call `record_step` after a completed optimizer step. Pass detached loss values
when possible. The observer does not install hooks, modify model state, retain
batches, or change gradients.

## Interpretation and limits

The report separates observable runtime facts from estimates. Parameter count
does not establish capability; task-specific benchmarks are required. A raw
softmax score is not automatically calibrated confidence, so this module does
not infer confidence or claim that the model knows its own limitations. External
hardware availability and operating-system metrics are outside this module's
scope.
