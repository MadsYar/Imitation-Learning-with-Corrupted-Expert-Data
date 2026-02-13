# train_bc.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
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
# EPISODE LEVEL FILTERING
# -------------------------------------------------------------------
def episode_level_filter(data, return_threshold=200.0):
    """
    Keep only episodes whose return >= threshold.
    """
    obs = data["observations"]
    acts = data["actions"]
    returns = data["returns"]
    lengths = data["episode_lengths"]

    keep_idx = []
    idx = 0
    for ret, length in zip(returns, lengths):
        end = idx + length
        if ret >= return_threshold:
            keep_idx.extend(range(idx, end))
        idx = end

    keep_idx = np.array(keep_idx, dtype=np.int64)
    return obs[keep_idx], acts[keep_idx]


# -------------------------------------------------------------------
# DATASET KEY (for model naming)
# -------------------------------------------------------------------
def get_dataset_key(path):
    """
    Convert filenames like:
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
    # e.g. flip_p0.10_500_seed123 → flip_p0.10_500
    if "seed" in fname:
        return fname.split("_seed")[0]

    return fname


# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------
def train_bc(
    data_path,
    out_path,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    seed=123,
    filter_bad=False,
    return_threshold=200.0,
):
    make_dirs()
    set_seed(seed)
    device = get_device()

    print_progress(f"[BC] Training using device: {device}")

    # Load dataset
    data = np.load(data_path)
    obs = data["observations"]
    acts = data["actions"]

    # Optional filtering
    if filter_bad:
        if "returns" not in data or "episode_lengths" not in data:
            print_progress("[WARNING] Filtering requested but dataset has no episode info. Skipping filtering.")
        else:
            print_progress(f"[BC] Applying EPISODE filtering (return >= {return_threshold})")
            obs_f, acts_f = episode_level_filter(data, return_threshold)
            if len(acts_f) == 0:
                print_progress(f"[WARNING] Filtering removed ALL samples — skipping model for {data_path}")
                return
            obs, acts = obs_f, acts_f
            print_progress(f"[BC] Filtering kept {len(acts)} samples.")

    # Convert to tensors
    X = torch.tensor(obs, dtype=torch.float32)
    y = torch.tensor(acts, dtype=torch.long)

    obs_dim = X.shape[1]
    act_dim = int(y.max().item()) + 1

    # Build network
    model = BCNet(obs_dim, act_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    print_progress(f"[BC] Dataset loaded: N={len(X)}, ObsDim={obs_dim}, ActDim={act_dim}")
    print_progress(f"[BC] Training {epochs} epochs on {os.path.basename(data_path)}")

    # Train loop
    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(X)
        print_progress(f"[BC] Epoch {epoch+1}/{epochs} — Loss={avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), out_path)
    print_progress(f"[BC] Saved model → {out_path}")


# -------------------------------------------------------------------
# CLI ENTRY
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--filter_bad", action="store_true")
    parser.add_argument("--return_threshold", type=float, default=200.0)
    args = parser.parse_args()

    train_bc(
        args.data,
        args.out,
        epochs=args.epochs,
        seed=args.seed,
        filter_bad=args.filter_bad,
        return_threshold=args.return_threshold,
    )
