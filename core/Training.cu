import torch
from model_training.lmlm_training import TrainConfig, train

model = torch.nn.Linear(8, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()
loader = [(torch.randn(16, 8), torch.randn(16, 1)) for _ in range(4)]
result = train(model, loader, optimizer, loss_fn,
               config=TrainConfig(epochs=2, device="cpu", precision="fp32"))
print(result.history)
