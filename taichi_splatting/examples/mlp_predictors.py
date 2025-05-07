import torch.nn as nn


class LogScalingMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, positions):
        return self.net(positions)


class AlphaMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, positions):
        return self.net(positions)