import random
from pathlib import Path

import numpy as np
import torch

from components.envs import make_env
from data.path_collector import MdpPathCollector
from data.replay_buffer import ReplayBuffer
from framework.checkpoint import save_checkpoint, save_config
from framework.logging import ProgressLogger
from framework.trainer import SACTrainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchBatchRLAlgorithm:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = make_env(cfg.env, cfg.seed)
        eval_render_mode = "human" if cfg.render_eval else None
        self.eval_env = make_env(cfg.env, cfg.seed + 1000, render_mode=eval_render_mode)
        self.output_dir = Path(cfg.output_dir) / cfg.experiment_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logger = ProgressLogger(self.output_dir)

        self.obs_dim = int(np.prod(self.env.observation_space.shape))
        self.action_dim = int(np.prod(self.env.action_space.shape))

        self.replay_buffer = ReplayBuffer(self.obs_dim, self.action_dim, cfg.replay_size)
        self.trainer = SACTrainer(self.obs_dim, self.action_dim, cfg, self.device)
        self.exploration_collector = MdpPathCollector(
            self.env,
            self.trainer.policy,
            max_path_length=cfg.max_episode_steps,
        )
        self.evaluation_collector = MdpPathCollector(
            self.eval_env,
            self.trainer.policy,
            max_path_length=cfg.max_episode_steps,
        )
        save_config(self.output_dir / "config.json", cfg)

    def train(self):
        if self.cfg.initial_random_steps > 0:
            initial_paths = self.exploration_collector.collect_new_paths(
                num_steps=self.cfg.initial_random_steps,
                random_actions=True,
            )
            self.replay_buffer.add_paths(initial_paths)

        for epoch in range(1, self.cfg.epochs + 1):
            expl_paths = self.exploration_collector.collect_new_paths(
                num_steps=self.cfg.steps_per_epoch,
                random_actions=False,
                deterministic=False,
            )
            self.replay_buffer.add_paths(expl_paths)

            last_stats = self.train_epoch()
            row = self.log_epoch(epoch, expl_paths, last_stats)
            self.logger.record(row)
            if self.cfg.plot_progress:
                self.logger.maybe_plot()
            self.save_models(epoch)

        self.env.close()
        self.eval_env.close()
        self.save_models("final")
        print(f"saved progress to {self.logger.csv_path}")
        print(f"saved final checkpoint to {self.checkpoint_dir / 'latest.pt'}")

    def train_epoch(self):
        last_stats = {}
        if self.replay_buffer.size < self.cfg.batch_size:
            return last_stats

        for _ in range(self.cfg.steps_per_epoch):
            batch = self.replay_buffer.sample(self.cfg.batch_size, self.device)
            last_stats = self.trainer.train_step(batch)
        return last_stats

    def log_epoch(self, epoch: int, expl_paths, last_stats):
        eval_paths = self.evaluation_collector.collect_new_paths(
            num_paths=self.cfg.eval_episodes,
            random_actions=False,
            deterministic=True,
        )
        avg_return = self.average_return(eval_paths)
        expl_return = self.average_return(expl_paths)
        eval_reward = self.average_step_reward(eval_paths)
        expl_reward = self.average_step_reward(expl_paths)
        expl_steps = sum(len(path["rewards"]) for path in expl_paths)
        stats_text = " ".join(f"{key}={value:.4f}" for key, value in last_stats.items())
        print(
            f"epoch={epoch:03d} expl_steps={expl_steps} "
            f"expl_return={expl_return:.2f} eval_return={avg_return:.2f} "
            f"expl_reward={expl_reward:.3f} eval_reward={eval_reward:.3f} "
            f"replay_size={self.replay_buffer.size} {stats_text}"
        )
        row = {
            "epoch": epoch,
            "expl_steps": expl_steps,
            "expl_return_mean": expl_return,
            "eval_return_mean": avg_return,
            "expl_reward_mean": expl_reward,
            "eval_reward_mean": eval_reward,
            "replay_size": self.replay_buffer.size,
        }
        row.update(last_stats)
        return row

    @staticmethod
    def average_return(paths):
        returns = [float(np.sum(path["rewards"])) for path in paths]
        return float(np.mean(returns)) if returns else 0.0

    @staticmethod
    def average_step_reward(paths):
        rewards = [path["rewards"] for path in paths if len(path["rewards"]) > 0]
        if not rewards:
            return 0.0
        return float(np.mean(np.concatenate(rewards, axis=0)))

    def save_models(self, epoch):
        save_checkpoint(
            self.checkpoint_dir / "latest.pt",
            self.trainer,
            self.cfg,
            self.obs_dim,
            self.action_dim,
            epoch,
        )
        if epoch == "final":
            save_checkpoint(
                self.checkpoint_dir / "final.pt",
                self.trainer,
                self.cfg,
                self.obs_dim,
                self.action_dim,
                epoch,
            )
        elif self.cfg.save_every > 0 and epoch % self.cfg.save_every == 0:
            save_checkpoint(
                self.checkpoint_dir / f"epoch_{epoch:04d}.pt",
                self.trainer,
                self.cfg,
                self.obs_dim,
                self.action_dim,
                epoch,
            )


def train(cfg):
    set_seed(cfg.seed)
    algorithm = TorchBatchRLAlgorithm(cfg)
    algorithm.train()
