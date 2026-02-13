# evaluate.py
import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import DQN

from utils import (
    make_dirs,
    set_seed,
    get_device,
    print_progress,
)


# -------------------------------------------------------------------
# NETWORK (BC/DAgger)
# -------------------------------------------------------------------
class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------
# MODEL TYPE DETECTION
# -------------------------------------------------------------------
def get_model_type(model_path):
    """
    Expert models are .zip,
    BC and DAgger models are .pt.
    """
    if model_path.endswith(".zip"):
        return "expert"
    fname = os.path.basename(model_path)
    if fname.startswith("bc_"):
        return "bc"
    if fname.startswith("dagger_"):
        return "dagger"
    return "unknown"


def get_dataset_key_from_model(model_path):
    """
    Examples:
        bc_seed0_clean500.pt → clean500
        dagger_seed1_flip_p0.10_500.pt → flip_p0.10_500
    """
    fname = os.path.basename(model_path).replace(".pt", "")
    parts = fname.split("_")

    # structure: algo, seedX, datasetkey
    if len(parts) < 3:
        return "unknown"

    dataset_key = "_".join(parts[2:])
    return dataset_key


# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
def load_model(model_path, env, device):
    model_type = get_model_type(model_path)

    if model_path.endswith(".zip"):
        print_progress(f"[Eval] Loading expert model {model_path}")
        model = DQN.load(model_path, env=env, device="cpu")
        return model, "expert"

    print_progress(f"[Eval] Loading policy model {model_path}")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    model = BCNet(obs_dim, act_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, model_type


# -------------------------------------------------------------------
# RUN EVALUATION
# -------------------------------------------------------------------
def evaluate_model(
    model_path: str,
    episodes: int = 50,
    seed: int = 123,
    success_threshold: float = 250,
):
    make_dirs()
    set_seed(seed)
    device = get_device()

    env = gym.make("CartPole-v1")
    model, model_type = load_model(model_path, env, device)

    rewards = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0

        while not done:
            if model_type == "expert":
                action, _ = model.predict(obs, deterministic=True)
            else:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = model(obs_t)
                action = torch.argmax(logits, dim=1).item()

            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r

        rewards.append(total)

    env.close()

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    success_rate = float(np.mean(np.array(rewards) >= success_threshold) * 100.0)

    dataset_key = "expert" if model_type == "expert" else get_dataset_key_from_model(model_path)

    print_progress(f"[Eval] {os.path.basename(model_path)} | "
                   f"{model_type.upper()} | "
                   f"mean={mean_r:.1f}, std={std_r:.1f}, success={success_rate:.1f}%")

    # Write results
    row = [
        os.path.basename(model_path),
        model_type,
        dataset_key,
        mean_r,
        std_r,
        success_rate,
        seed,
    ]

    with open("logs/evaluation_fixed.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return mean_r, std_r, success_rate


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--threshold", type=float, default=250)
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        episodes=args.episodes,
        seed=args.seed,
        success_threshold=args.threshold,
    )
