# model.py
"""
DemandNet: small MLP that predicts units sold in the next 7 days for an
e-commerce listing (standardised target, see ecommerce_scaler.json).

The architecture is intentionally tiny (~3.2k parameters) so the exact same
network can be trained in plain JavaScript on a phone (phone_client/app/trainer.js).
Parameter names/shapes must stay identical across PyTorch and JS:
    fc1.weight [64, F]  fc1.bias [64]
    fc2.weight [32, 64] fc2.bias [32]
    fc3.weight [1, 32]  fc3.bias [1]
"""
import torch
import torch.nn as nn

NUM_FEATURES = 16


class Model(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)
