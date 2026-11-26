import torch
import pytest

from model_training.lmlm_training import (
    CheckpointManager, TrainConfig, benchmark, evaluate, load_checkpoint,
    resolve_device_plan, train,
)


def make_problem():
    torch.manual_seed(7)
    x = torch.randn(32, 4)
    y = x.sum(dim=1, keepdim=True)
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y), batch_size=8, shuffle=False
    )


def test_cpu_training_and_evaluation():
    loader = make_problem()
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = evaluate(model, loader, torch.nn.MSELoss())["loss"]
    result = train(
        model, loader, optimizer, torch.nn.MSELoss(),
        config=TrainConfig(epochs=3, accumulation_steps=2, device="cpu"),
        validation_loader=loader,
    )
    after = evaluate(model, loader, torch.nn.MSELoss())["loss"]
    assert result.global_step == 6
    assert after < before
    assert result.history[-1]["validation_loss"] >= 0


def test_checkpoint_round_trip(tmp_path):
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    manager = CheckpointManager(tmp_path)
    manager.save("state.pt", model=model, optimizer=optimizer, epoch=2, global_step=9)
    restored = torch.nn.Linear(4, 1)
    state = load_checkpoint(tmp_path / "state.pt", model=restored)
    for a, b in zip(model.parameters(), restored.parameters()):
        assert torch.equal(a, b)
    assert state["epoch"] == 2
    assert state["global_step"] == 9


def test_checkpoint_rejects_path_traversal(tmp_path):
    manager = CheckpointManager(tmp_path)
    with pytest.raises(ValueError):
        manager.save("../escape.pt", model=torch.nn.Linear(1, 1),
                     epoch=0, global_step=0)


def test_cpu_device_plan_is_conservative():
    plan = resolve_device_plan("cpu", "auto")
    assert plan.device.type == "cpu"
    assert plan.precision == "fp32"
    assert not plan.amp_enabled


def test_benchmark_reports_positive_latency():
    model = torch.nn.Linear(4, 1)
    result = benchmark(model, torch.randn(2, 4), device="cpu",
                      precision="fp32", warmup=1, iterations=3)
    assert result.batch_size == 2
    assert result.mean_latency_ms > 0
    assert result.samples_per_second > 0
