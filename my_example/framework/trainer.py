import torch
import torch.nn as nn
import torch.nn.functional as F

from components.networks import QFunction
from components.policies import TanhGaussianPolicy


class SACTrainer:
    def __init__(self, obs_dim: int, action_dim: int, cfg, device: torch.device):
        hidden_sizes = (cfg.hidden_size, cfg.hidden_size)
        self.cfg = cfg
        self.device = device
        self.policy = TanhGaussianPolicy(obs_dim, action_dim, hidden_sizes).to(device)
        self.q1 = QFunction(obs_dim, action_dim, hidden_sizes).to(device)
        self.q2 = QFunction(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_q1 = QFunction(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_q2 = QFunction(obs_dim, action_dim, hidden_sizes).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=cfg.lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=cfg.lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train_step(self, batch):
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        q1_loss, q2_loss = self.update_q_functions(obs, actions, rewards, next_obs, dones)
        policy_loss, log_prob = self.update_policy(obs)
        alpha_loss = self.update_alpha(log_prob)
        self.soft_update(self.q1, self.target_q1)
        self.soft_update(self.q2, self.target_q2)

        return {
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha.item(),
        }

    def update_q_functions(self, obs, actions, rewards, next_obs, dones):
        with torch.no_grad():
            next_actions, next_log_prob = self.policy.sample(next_obs)
            next_q = torch.min(
                self.target_q1(next_obs, next_actions),
                self.target_q2(next_obs, next_actions),
            )
            q_target = rewards + (1 - dones) * self.cfg.discount * (
                next_q - self.alpha.detach() * next_log_prob
            )

        q1_loss = F.mse_loss(self.q1(obs, actions), q_target)
        q2_loss = F.mse_loss(self.q2(obs, actions), q_target)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        return q1_loss.item(), q2_loss.item()

    def update_policy(self, obs):
        new_actions, log_prob = self.policy.sample(obs)
        q_new_actions = torch.min(self.q1(obs, new_actions), self.q2(obs, new_actions))
        policy_loss = (self.alpha.detach() * log_prob - q_new_actions).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()
        return policy_loss.item(), log_prob

    def update_alpha(self, log_prob):
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        return alpha_loss.item()

    def soft_update(self, source: nn.Module, target: nn.Module):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1 - self.cfg.tau)
            target_param.data.add_(self.cfg.tau * source_param.data)
