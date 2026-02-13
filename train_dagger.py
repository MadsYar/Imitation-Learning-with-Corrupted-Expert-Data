# train_dagger.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3 import DQN
import gymnasium as gym

from utils import make_dirs, set_seed, get_device, print_progress


# -------------------------------------------------------------------
# NETWORK
# -------------------------------------------------------------------
class BCNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------------------------------------------------
# DATASET KEY (same scheme as BC)
# -------------------------------------------------------------------
def get_dataset_key(path):
    """
    Convert dataset filename to model naming key.
    Examples:

        clean_500_seed123.npz → clean500
        clean_500_seed123_step.npz → clean500_step
        flip_p0.10_500_seed123.npz → flip_p0.10_500
    """
    fname = os.path.basename(path).replace(".npz", "")

    # Clean datasets
    if fname.startswith("clean_"):
        parts = fname.split("_")
        N = parts[1]
        if "step" in fname:
            return f"clean{N}_step"
        if "epf" in fname:
            return f"clean{N}_epf"
        return f"clean{N}"

    # Noisy datasets
    if "seed" in fname:
        return fname.split("_seed")[0]  # e.g. flip_p0.10_500

    return fname


# -------------------------------------------------------------------
# DATA LOADER
# -------------------------------------------------------------------
def make_dataloader(obs, acts, batch_size):
    X = torch.from_numpy(obs).float()
    y = torch.from_numpy(acts).long()
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


# -------------------------------------------------------------------
# TRAIN ONE EPOCH
# -------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(xb)
        count += len(xb)

    return total_loss / max(count, 1)


# -------------------------------------------------------------------
# COLLECT DAGGER DATA (ask expert for correct action)
# -------------------------------------------------------------------
def collect_dagger_data(env, model, expert, episodes, device):
    model.eval()

    new_obs = []
    new_acts = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not (done or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(obs_t)
                student_action = torch.argmax(logits, dim=1).item()

            expert_action, _ = expert.predict(obs, deterministic=True)

            new_obs.append(obs.copy())
            new_acts.append(int(expert_action))

            obs, _, done, truncated, _ = env.step(student_action)

    return (
        np.array(new_obs, dtype=np.float32),
        np.array(new_acts, dtype=np.int64),
    )


# -------------------------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------------------------
def train_dagger(
    expert_path: str,
    init_dataset_path: str,
    out_model_path: str,
    iterations: int = 3,
    episodes_per_iter: int = 5,
    epochs_per_iter: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 123,
):
    make_dirs()
    set_seed(seed)
    device = get_device()

    print_progress(f"[DAgger] Training using device: {device}")

    # Load environment + expert
    env = gym.make("CartPole-v1")
    expert = DQN.load(expert_path, env=env, device="cpu")

    # Load initial dataset
    data = np.load(init_dataset_path)
    obs = data["observations"].astype(np.float32)
    acts = data["actions"].astype(np.int64)

    obs_dim = obs.shape[1]
    act_dim = int(acts.max()) + 1

    print_progress(f"[DAgger] Loaded initial dataset: {init_dataset_path}")
    print_progress(f"[DAgger] Initial size: {len(obs)}")

    # Create student model
    model = BCNet(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # DAgger iterations
    for it in range(iterations):
        print_progress(f"[DAgger] Iteration {it+1}/{iterations}")

        # 1) Train on current dataset
        loader = make_dataloader(obs, acts, batch_size)

        for ep in range(epochs_per_iter):
            loss = train_one_epoch(model, loader, optimizer, device)
            print_progress(f"  Epoch {ep+1}/{epochs_per_iter}, loss={loss:.4f}")

        # 2) Collect new states & expert labels
        new_obs, new_acts = collect_dagger_data(
            env, model, expert, episodes_per_iter, device
        )

        print_progress(f"  Collected {len(new_obs)} new samples")

        # 3) Aggregate dataset
        obs = np.concatenate([obs, new_obs], axis=0)
        acts = np.concatenate([acts, new_acts], axis=0)

        print_progress(f"  Dataset size now {len(obs)}")

    # Save final model
    torch.save(model.state_dict(), out_model_path)
    print_progress(f"[DAgger] Saved model → {out_model_path}")

    env.close()


# -------------------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes_per_iter", type=int, default=5)
    parser.add_argument("--epochs_per_iter", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    train_dagger(
        expert_path=args.expert,
        init_dataset_path=args.data,
        out_model_path=args.out,
        iterations=args.iterations,
        episodes_per_iter=args.episodes_per_iter,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
