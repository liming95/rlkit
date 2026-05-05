import argparse
from pathlib import Path

import numpy as np
import torch

from components.envs import make_env, reset_env, scale_action, step_env
from framework.checkpoint import load_policy


def run_policy(args):
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train first, or pass the checkpoint path printed after training.\n"
            "Current default training command:\n"
            "  python sac.py --env TinyContinuousCartPole-v0 --epochs 20 --experiment-name sac_cartpole\n"
            "Current default inference command:\n"
            "  python infer.py --checkpoint runs/sac_cartpole/checkpoints/latest.pt\n"
            "For older runs made before the cart-pole rename, try:\n"
            "  python infer.py --checkpoint runs/sac_pendulum/checkpoints/latest.pt"
        )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    policy, checkpoint = load_policy(checkpoint_path, device)
    cfg = checkpoint["config"]

    env_name = args.env or cfg["env"]
    render_mode = "human" if args.render else None
    env = make_env(env_name, args.seed, render_mode=render_mode)
    action_low = env.action_space.low
    action_high = env.action_space.high

    returns = []
    for episode in range(1, args.episodes + 1):
        obs = reset_env(env)
        episode_return = 0.0

        for _ in range(args.max_episode_steps or cfg["max_episode_steps"]):
            normalized_action = policy.act(obs, deterministic=True, device=device)
            env_action = scale_action(normalized_action, action_low, action_high)
            obs, reward, done, _ = step_env(env, env_action)
            episode_return += reward
            if done:
                break

        returns.append(episode_return)
        print(f"episode={episode:03d} return={episode_return:.2f}")

    print(f"mean_return={float(np.mean(returns)):.2f}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a saved SAC policy.")
    parser.add_argument(
        "--checkpoint",
        default="runs/sac_cartpole/checkpoints/latest.pt",
        help="Path to a checkpoint saved by sac.py.",
    )
    parser.add_argument("--env", default=None, help="Override checkpoint environment.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_policy(parse_args())
