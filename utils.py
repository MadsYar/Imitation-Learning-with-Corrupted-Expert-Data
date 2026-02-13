# utils.py
import os
import random
import numpy as np
import torch


# -------------------------------------------------------------------
# DIRECTORY SETUP
# -------------------------------------------------------------------
def make_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/expert", exist_ok=True)
    os.makedirs("data/noisy", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/baselines", exist_ok=True)
    os.makedirs("plots/noise", exist_ok=True)
    os.makedirs("plots/filtering", exist_ok=True)
    os.makedirs("plots/confidence", exist_ok=True)


# -------------------------------------------------------------------
# SEEDING
# -------------------------------------------------------------------
def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        import gymnasium as gym
        gym.utils.seeding.np_random(seed)
    except Exception:
        pass


# -------------------------------------------------------------------
# DEVICE
# -------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------------------
# PRINTING
# -------------------------------------------------------------------
def print_progress(msg: str):
    print(f"[INFO] {msg}")
