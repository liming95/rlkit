import json
from dataclasses import asdict
from pathlib import Path

import torch

from components.policies import TanhGaussianPolicy


def save_checkpoint(path, trainer, cfg, obs_dim, action_dim, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "config": asdict(cfg),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "policy_state_dict": trainer.policy.state_dict(),
        "q1_state_dict": trainer.q1.state_dict(),
        "q2_state_dict": trainer.q2.state_dict(),
        "target_q1_state_dict": trainer.target_q1.state_dict(),
        "target_q2_state_dict": trainer.target_q2.state_dict(),
        "policy_optimizer_state_dict": trainer.policy_opt.state_dict(),
        "q1_optimizer_state_dict": trainer.q1_opt.state_dict(),
        "q2_optimizer_state_dict": trainer.q2_opt.state_dict(),
        "alpha_optimizer_state_dict": trainer.alpha_opt.state_dict(),
        "log_alpha": trainer.log_alpha.detach().cpu(),
    }
    torch.save(payload, path)


def save_config(path, cfg):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def load_policy(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    hidden_sizes = (cfg["hidden_size"], cfg["hidden_size"])
    policy = TanhGaussianPolicy(
        checkpoint["obs_dim"],
        checkpoint["action_dim"],
        hidden_sizes,
    ).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
    return policy, checkpoint
