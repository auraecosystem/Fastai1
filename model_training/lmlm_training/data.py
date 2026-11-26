"""Composable PyTorch data-loading adapters for Fastai1.

This module is independent of fastai's internal package. It accepts ordinary
PyTorch Dataset/IterableDataset instances and leaves dataset semantics,
tokenization, and model-specific collation to the caller.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

Transform = Callable[[Any], Any]


def _apply(transform: Transform | Sequence[Transform] | None, value: Any) -> Any:
    """Apply one transform or a sequence of transforms in order."""
    if transform is None:
        return value
    if isinstance(transform, Sequence) and not isinstance(transform, (str, bytes)):
        for fn in transform:
            value = fn(value)
        return value
    return transform(value)


def move_to_device(value: Any, device: str | torch.device | None) -> Any:
    """Move tensors nested in common containers without changing other values."""
    if device is None:
        return value
    target = torch.device(device)
    if isinstance(value, torch.Tensor):
        return value.to(target, non_blocking=target.type == "cuda")
    if isinstance(value, Mapping):
        items = [(key, move_to_device(item, target)) for key, item in value.items()]
        try:
            return type(value)(items)
        except (TypeError, ValueError):
            return dict(items)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(move_to_device(item, target) for item in value))
    if isinstance(value, tuple):
        return tuple(move_to_device(item, target) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, target) for item in value]
    return value


class _MapTransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return _apply(self.transform, self.dataset[index])

    def __getattr__(self, name):
        if name in {"dataset", "transform"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)


class _IterableTransformDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        for item in self.dataset:
            yield _apply(self.transform, item)


class _BeforeBatchCollator:
    """Picklable collator that applies before_batch before PyTorch collation."""

    def __init__(self, collate_fn, transform):
        self.collate_fn = collate_fn
        self.transform = transform

    def __call__(self, samples):
        return self.collate_fn(_apply(self.transform, samples))


class DataLoaderAdapter:
    """Transform-aware wrapper around torch.utils.data.DataLoader.

    Transform order:
      1. after_item runs on each dataset item.
      2. before_batch runs on the list of items before collation.
      3. PyTorch collate_fn creates the batch.
      4. Tensor values are moved to device, when configured.
      5. after_batch runs on the resulting batch.

    Device transfer and after_batch execute in the main training process.
    """

    def __init__(
        self,
        dataset,
        *,
        batch_size: int = 64,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        collate_fn=None,
        after_item=None,
        before_batch=None,
        after_batch=None,
        device: str | torch.device | None = None,
        **loader_kwargs,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        self.dataset = dataset
        self.device = torch.device(device) if device is not None else None
        self.after_item = after_item
        self.before_batch = before_batch
        self.after_batch = after_batch
        self._config = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": drop_last,
            "collate_fn": collate_fn,
            **loader_kwargs,
        }

        if isinstance(dataset, IterableDataset):
            transformed_dataset = _IterableTransformDataset(dataset, after_item)
        else:
            transformed_dataset = _MapTransformDataset(dataset, after_item)

        base_collate = collate_fn or torch.utils.data.default_collate
        collator = _BeforeBatchCollator(base_collate, before_batch)
        self._loader = DataLoader(
            transformed_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collator,
            **loader_kwargs,
        )

    def __iter__(self):
        for batch in self._loader:
            batch = move_to_device(batch, self.device)
            yield _apply(self.after_batch, batch)

    def __len__(self):
        return len(self._loader)

    def __getattr__(self, name):
        if name in {"_loader", "_config"}:
            raise AttributeError(name)
        return getattr(self._loader, name)

    def to(self, device: str | torch.device | None):
        self.device = torch.device(device) if device is not None else None
        return self

    def new(self, dataset=None, **overrides):
        """Create a loader for another dataset while preserving its settings."""
        config = {**self._config, **overrides}
        return type(self)(
            self.dataset if dataset is None else dataset,
            after_item=self.after_item,
            before_batch=self.before_batch,
            after_batch=self.after_batch,
            device=self.device,
            **config,
        )


class DataLoaders:
    """Container for training and validation DataLoaderAdapter objects."""

    def __init__(self, train: DataLoaderAdapter, valid: DataLoaderAdapter):
        self.train = train
        self.valid = valid
        self.device = train.device

    def __getitem__(self, index):
        if index == 0:
            return self.train
        if index == 1:
            return self.valid
        raise IndexError(index)

    def __len__(self):
        return 2

    def to(self, device: str | torch.device | None):
        self.device = torch.device(device) if device is not None else None
        self.train.to(self.device)
        self.valid.to(self.device)
        return self

    def test_loader(self, dataset, **overrides):
        """Build a non-shuffling test loader using validation transforms."""
        config = {"shuffle": False, "drop_last": False, **overrides}
        return self.valid.new(dataset, **config)

    @classmethod
    def from_datasets(
        cls,
        train_dataset,
        valid_dataset,
        *,
        batch_size: int = 64,
        device: str | torch.device | None = None,
        train_kwargs: dict[str, Any] | None = None,
        valid_kwargs: dict[str, Any] | None = None,
    ):
        train_options = {"shuffle": True, **(train_kwargs or {})}
        valid_options = {"shuffle": False, "drop_last": False, **(valid_kwargs or {})}
        train = DataLoaderAdapter(
            train_dataset, batch_size=batch_size, device=device, **train_options
        )
        valid = DataLoaderAdapter(
            valid_dataset, batch_size=batch_size, device=device, **valid_options
        )
        return cls(train, valid)


def make_dataloaders(
    train_dataset,
    valid_dataset,
    *,
    batch_size: int = 64,
    device: str | torch.device | None = None,
    train_kwargs: dict[str, Any] | None = None,
    valid_kwargs: dict[str, Any] | None = None,
) -> DataLoaders:
    """Convenience factory for a training/validation loader pair."""
    return DataLoaders.from_datasets(
        train_dataset,
        valid_dataset,
        batch_size=batch_size,
        device=device,
        train_kwargs=train_kwargs,
        valid_kwargs=valid_kwargs,
    )


__all__ = ["DataLoaderAdapter", "DataLoaders", "make_dataloaders", "move_to_device"]
