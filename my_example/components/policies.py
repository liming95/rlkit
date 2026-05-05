import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from components.networks import MLP


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class TanhGaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes):
        super().__init__()
        self.backbone = MLP(obs_dim, hidden_sizes[-1], hidden_sizes[:-1])
        self.mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs):
        h = self.backbone(obs)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + EPSILON)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs, deterministic: bool, device=None):
        if device is None:
            device = next(self.parameters()).device
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mean, log_std = self(obs_t)
        if deterministic:
            action = torch.tanh(mean)
        else:
            action = torch.tanh(Normal(mean, log_std.exp()).sample())
        return action.cpu().numpy()[0].astype(np.float32)
