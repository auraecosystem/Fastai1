import torch
import pytest

from model_training.self_awareness import (
    ModelIntrospector,
    SelfAwareness,
    TrainingAwareness,
)


def test_model_snapshot_counts_parameters_and_modules():
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU())
    model[0].bias.requires_grad_(False)

    snapshot = ModelIntrospector.inspect(model)

    assert snapshot.total_parameters == 15
    assert snapshot.trainable_parameters == 12
    assert snapshot.frozen_parameters == 3
    assert snapshot.module_count == 3
    assert snapshot.training is True
    assert "cpu" in snapshot.devices


def test_tensor_health_detects_non_finite_values():
    tensor = torch.tensor([1.0, float("nan"), 3.0])
    health = ModelIntrospector.inspect_tensor(tensor)

    assert health.finite is False
    assert health.finite_fraction == pytest.approx(2 / 3)
    assert health.mean == pytest.approx(2.0)


def test_training_awareness_tracks_steps_and_metrics():
    tracker = TrainingAwareness(history_size=2)
    tracker.start_epoch()
    tracker.record_step(torch.tensor(3.0), batch_size=4, metrics={"accuracy": 0.5})
    snapshot = tracker.record_step(1.0, batch_size=5)

    assert snapshot.epoch == 1
    assert snapshot.global_step == 2
    assert snapshot.samples_seen == 9
    assert snapshot.mean_recent_loss == pytest.approx(2.0)
    assert snapshot.best_loss == pytest.approx(1.0)
    assert snapshot.latest_metrics == {"accuracy": 0.5}


def test_non_finite_loss_is_counted_not_added_to_average():
    tracker = TrainingAwareness()
    tracker.record_step(float("nan"))
    snapshot = tracker.record_step(2.0)

    assert snapshot.non_finite_loss_count == 1
    assert snapshot.mean_recent_loss == pytest.approx(2.0)
    assert snapshot.loss_is_finite is True


def test_self_awareness_report_is_json_serializable():
    awareness = SelfAwareness(torch.nn.Linear(2, 1))
    awareness.record_step(0.25, batch_size=8)
    report = awareness.inspect()

    assert report["scope"] == "observable_runtime_state"
    assert report["model"]["total_parameters"] == 3
    assert report["training"]["samples_seen"] == 8
    assert "limitations" in report
