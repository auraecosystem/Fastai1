import torch

from model_training.lmlm_training.trainer import TrainConfig, train
from model_training.self_awareness import SelfAwareness


def _make_training_objects():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batches = [
        (torch.tensor([[1.0], [2.0]]), torch.tensor([[2.0], [4.0]])),
        (torch.tensor([[3.0]]), torch.tensor([[6.0]])),
        (torch.tensor([[4.0], [5.0]]), torch.tensor([[8.0], [10.0]])),
    ]
    return model, optimizer, batches


def test_trainer_records_completed_accumulated_optimizer_updates():
    model, optimizer, batches = _make_training_objects()
    observer = SelfAwareness(model)
    result = train(
        model, batches, optimizer, torch.nn.functional.mse_loss,
        config=TrainConfig(epochs=2, accumulation_steps=2, device="cpu", precision="fp32"),
        self_awareness=observer,
    )
    report = observer.inspect()["training"]
    assert result.global_step == 4
    assert report["global_step"] == 4
    assert report["epoch"] == 2
    assert report["samples_seen"] == 10
    assert report["latest_loss"] is not None
    assert report["loss_is_finite"] is True


def test_trainer_without_observer_remains_supported():
    model, optimizer, batches = _make_training_objects()
    result = train(
        model, batches, optimizer, torch.nn.functional.mse_loss,
        config=TrainConfig(epochs=1, accumulation_steps=2, device="cpu", precision="fp32"),
    )
    assert result.global_step == 2
    assert result.history[0]["optimizer_steps"] == 2.0


def test_trainer_rejects_invalid_observer():
    model, optimizer, batches = _make_training_objects()
    try:
        train(model, batches, optimizer, torch.nn.functional.mse_loss,
              config=TrainConfig(device="cpu", precision="fp32"), self_awareness=object())
    except TypeError as exc:
        assert "start_epoch" in str(exc)
    else:
        raise AssertionError("invalid observer should be rejected")
