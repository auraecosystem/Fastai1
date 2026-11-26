import torch
from torch.utils.data import TensorDataset

from model_training.lmlm_training.data import (
    DataLoaderAdapter,
    DataLoaders,
    make_dataloaders,
    move_to_device,
)


def _dataset(n=10):
    return TensorDataset(torch.arange(n, dtype=torch.float32).unsqueeze(1),
                         torch.arange(n, dtype=torch.float32))


def test_loader_applies_item_batch_and_after_batch_transforms():
    loader = DataLoaderAdapter(
        _dataset(4),
        batch_size=2,
        shuffle=False,
        after_item=lambda item: (item[0] + 1, item[1]),
        before_batch=lambda items: items,
        after_batch=lambda batch: (batch[0] * 2, batch[1]),
    )
    x, y = next(iter(loader))
    assert torch.equal(x, torch.tensor([[2.0], [4.0]]))
    assert torch.equal(y, torch.tensor([0.0, 1.0]))


def test_loader_keeps_last_partial_batch_by_default():
    loader = DataLoaderAdapter(_dataset(5), batch_size=2)
    assert len(loader) == 3
    assert [len(batch[1]) for batch in loader] == [2, 2, 1]


def test_new_loader_preserves_transforms_and_settings():
    loader = DataLoaderAdapter(
        _dataset(4), batch_size=2, shuffle=False,
        after_batch=lambda batch: (batch[0] + 3, batch[1]),
    )
    replacement = loader.new(_dataset(2))
    x, _ = next(iter(replacement))
    assert torch.equal(x, torch.tensor([[3.0], [4.0]]))
    assert replacement.batch_size == 2


def test_dataloaders_build_train_valid_and_test_loaders():
    dls = make_dataloaders(_dataset(8), _dataset(4), batch_size=2)
    assert isinstance(dls, DataLoaders)
    assert len(dls) == 2
    assert len(dls.train) == 4
    assert len(dls.valid) == 2
    test_dl = dls.test_loader(_dataset(3))
    assert len(test_dl) == 2
    assert [len(batch[1]) for batch in test_dl] == [2, 1]


def test_move_to_device_recurses_and_preserves_non_tensors():
    batch = {"x": torch.ones(2), "meta": ("sample", 3)}
    moved = move_to_device(batch, "cpu")
    assert moved["x"].device.type == "cpu"
    assert moved["meta"] == ("sample", 3)


def test_invalid_batch_size_is_rejected():
    try:
        DataLoaderAdapter(_dataset(), batch_size=0)
    except ValueError as exc:
        assert "batch_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")
