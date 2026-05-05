import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_path(self, path):
        for obs, action, reward, next_obs, done in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
        ):
            self.add(obs, action, reward, next_obs, done)

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.as_tensor(self.obs[idxs], device=device),
            "actions": torch.as_tensor(self.actions[idxs], device=device),
            "rewards": torch.as_tensor(self.rewards[idxs], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idxs], device=device),
            "dones": torch.as_tensor(self.dones[idxs], device=device),
        }
