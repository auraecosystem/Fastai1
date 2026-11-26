# LMLM training utilities

This is the historical implementation package for Fastai1's generic native PyTorch training adapter. Prefer importing its public API from 'model_training.lmlm_training' (lowercase) so imports work consistently on case-sensitive filesystems.

The package accepts an existing 'torch.nn.Module'; it does not own LMLM architecture, tokenization, datasets, or model artifacts. See docs/Training.md for usage, precision constraints, checkpoint trust warnings, and current limitations.
