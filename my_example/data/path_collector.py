import numpy as np

from components.envs import normalize_action, reset_env, scale_action, step_env


class MdpPathCollector:
    def __init__(self, env, policy=None, max_path_length=1000):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.num_steps_total = 0
        self.num_paths_total = 0

    def collect_new_paths(
        self,
        num_steps=None,
        num_paths=None,
        random_actions=False,
        deterministic=False,
    ):
        if num_steps is None and num_paths is None:
            raise ValueError("Either num_steps or num_paths must be provided.")

        paths = []
        steps_collected = 0

        while self._should_collect_more(paths, steps_collected, num_steps, num_paths):
            path = self.rollout(random_actions=random_actions, deterministic=deterministic)
            paths.append(path)
            path_len = len(path["rewards"])
            steps_collected += path_len
            self.num_steps_total += path_len
            self.num_paths_total += 1

        return paths

    @staticmethod
    def _should_collect_more(paths, steps_collected, num_steps, num_paths):
        if num_paths is not None and len(paths) >= num_paths:
            return False
        if num_steps is not None and steps_collected >= num_steps:
            return False
        return True

    def rollout(self, random_actions=False, deterministic=False):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []

        obs = reset_env(self.env)
        action_low = self.env.action_space.low
        action_high = self.env.action_space.high

        for _ in range(self.max_path_length):
            if random_actions:
                env_action = self.env.action_space.sample()
                buffer_action = normalize_action(env_action, action_low, action_high)
            else:
                normalized_action = self.policy.act(obs, deterministic=deterministic)
                env_action = scale_action(normalized_action, action_low, action_high)
                buffer_action = normalized_action

            next_obs, reward, done, _ = step_env(self.env, env_action)

            observations.append(obs)
            actions.append(buffer_action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(done)

            obs = next_obs
            if done:
                break

        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
            "next_observations": np.asarray(next_observations, dtype=np.float32),
            "terminals": np.asarray(terminals, dtype=np.float32).reshape(-1, 1),
        }

    def get_diagnostics(self):
        return {
            "num_steps_total": self.num_steps_total,
            "num_paths_total": self.num_paths_total,
        }
