# Fastai1

Fastai1 is an experimental, fastai-v1-inspired Python codebase containing legacy data/model utilities, Cython source modules, notebooks, and a native PyTorch training adapter for LMLM.

## Project status
This repository is being modernized incrementally. The legacy top-level modules and notebooks are retained; they are not yet a complete drop-in reimplementation of fastai v1. The supported modern entry point is currently 'model_training.lmlm_training'.

## Install
Python 3.10+ is supported for the modern training package.

```bash
python -m pip install -e ".[training,dev]"
```

For CPU-only environments, install a compatible PyTorch build separately if the default package index does not provide the desired wheel.

## Quick start
```python
import torch
from model_training.lmlm_training import TrainConfig, train

model = torch.nn.Linear(8, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
loader = [(torch.randn(16, 8), torch.randn(16, 1)) for _ in range(4)]
result = train(model, loader, optimizer, loss_fn,
               config=TrainConfig(epochs=2, device="cpu", precision="fp32"))
print(result.history)
```

## Architecture boundary
- Fastai1 provides reusable data/training utilities and compatibility surfaces.
- LMLM owns its model architecture, tokenizer, data semantics, model artifacts, and local runtime.
- The PyTorch adapter accepts an existing 'torch.nn.Module'; it does not define or download an LMLM model.
- Training and optimizer state are deliberately separate from inference/processor protocols.

See docs/Training.md for configuration, checkpoint safety, and known limitations.

## Development
```bash
python -m pip install -e ".[training,dev]"
pytest
```

The historical Cython files are retained as source. Building them is optional and requires a separate compatibility audit before they should be compiled or imported as production extensions.
