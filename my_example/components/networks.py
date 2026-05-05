import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes):
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(last_dim, hidden_size), nn.ReLU()])
            last_dim = hidden_size
        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class QFunction(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes):
        super().__init__()
        self.net = MLP(obs_dim + action_dim, 1, hidden_sizes)

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))
