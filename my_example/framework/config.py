import argparse
from dataclasses import dataclass


@dataclass
class SACConfig:
    env: str = "TinyContinuousCartPole-v0"
    seed: int = 0
    epochs: int = 10
    steps_per_epoch: int = 1000
    initial_random_steps: int = 1000
    eval_episodes: int = 5
    max_episode_steps: int = 200
    replay_size: int = 1_000_000
    batch_size: int = 256
    hidden_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    lr: float = 3e-4
    output_dir: str = "runs"
    experiment_name: str = "sac_cartpole"
    save_every: int = 1
    render_eval: bool = False
    plot_progress: bool = False


def parse_args() -> SACConfig:
    parser = argparse.ArgumentParser(description="Standalone Soft Actor-Critic")
    parser.add_argument("--env", default=SACConfig.env)
    parser.add_argument("--seed", type=int, default=SACConfig.seed)
    parser.add_argument("--epochs", type=int, default=SACConfig.epochs)
    parser.add_argument("--steps-per-epoch", type=int, default=SACConfig.steps_per_epoch)
    parser.add_argument("--initial-random-steps", type=int, default=SACConfig.initial_random_steps)
    parser.add_argument("--eval-episodes", type=int, default=SACConfig.eval_episodes)
    parser.add_argument("--max-episode-steps", type=int, default=SACConfig.max_episode_steps)
    parser.add_argument("--replay-size", type=int, default=SACConfig.replay_size)
    parser.add_argument("--batch-size", type=int, default=SACConfig.batch_size)
    parser.add_argument("--hidden-size", type=int, default=SACConfig.hidden_size)
    parser.add_argument("--discount", type=float, default=SACConfig.discount)
    parser.add_argument("--tau", type=float, default=SACConfig.tau)
    parser.add_argument("--lr", type=float, default=SACConfig.lr)
    parser.add_argument("--output-dir", default=SACConfig.output_dir)
    parser.add_argument("--experiment-name", default=SACConfig.experiment_name)
    parser.add_argument("--save-every", type=int, default=SACConfig.save_every)
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--plot-progress", action="store_true")
    args = parser.parse_args()
    return SACConfig(**vars(args))
