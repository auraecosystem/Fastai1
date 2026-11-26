# LMLM training utilities

This package provides a generic native PyTorch training adapter for an already
constructed `torch.nn.Module`. It does not own LMLM architecture, tokenization,
datasets, or model artifacts.

Use the stable lowercase import path:

```python
from model_training.lmlm_training import TrainConfig, train
```

The historical import `model_training.Lmlm_training` remains available through
a compatibility module. A module shim avoids case-only directory collisions
on Windows and case-insensitive macOS filesystems.

See [the training guide](../../docs/Training.md) for configuration, checkpoint
trust warnings, and current limitations.
