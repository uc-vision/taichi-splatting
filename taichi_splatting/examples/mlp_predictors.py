import torch.nn as nn


class CovarianceMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 2 for log-scaling, 2 for 2D rotation (unit complex number)
        )

    def forward(self, positions):
        return self.net(positions)


class AlphaMLP(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, positions):
        return self.net(positions)