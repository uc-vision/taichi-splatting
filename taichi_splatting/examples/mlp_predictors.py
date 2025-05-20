import torch.nn as nn


class CovarianceMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, latent):
        return self.net(latent)


class AlphaMLP(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, latent):
        return self.net(latent)