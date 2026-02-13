# noisy_data.py
import os
import argparse
import numpy as np

from utils import make_dirs, set_seed, print_progress


# -------------------------------------------------------------------
# NOISE FUNCTIONS
# -------------------------------------------------------------------
def add_flip_noise(actions, p):
    noisy = actions.copy()
    mask = np.random.rand(len(actions)) < p
    noisy[mask] = 1 - noisy[mask]
    return noisy


def add_gaussian_noise(observations, sigma):
    return observations + np.random.normal(0, sigma, size=observations.shape)


def add_dropout_noise(observations, p):
    noisy = observations.copy()
    mask = np.random.rand(*observations.shape) < p
    noisy[mask] = 0.0
    return noisy


def add_bias_noise(observations, bias_std):
    bias = np.random.normal(0.0, bias_std, size=(1, observations.shape[1]))
    return observations + bias


def add_jitter_noise(observations, p):
    noisy = observations.copy()
    mask = np.random.rand(len(observations)) < p
    for i in range(1, len(noisy)):
        if mask[i]:
            noisy[i] = noisy[i - 1]
    return noisy


# -------------------------------------------------------------------
# GENERATE NOISY DATASETS
# -------------------------------------------------------------------
def generate_noisy_datasets(
    clean_path: str,
    N: int,
    seed: int,
    flip_levels,
    gauss_levels,
    dropout_levels,
    bias_levels,
    jitter_levels,
):
    make_dirs()
    set_seed(seed)

    data = np.load(clean_path)
    obs = data["observations"]
    acts = data["actions"]

    returns = data.get("returns", None)
    lengths = data.get("episode_lengths", None)

    print_progress(f"Loaded clean dataset ({clean_path}) with {len(acts)} steps")

    outdir = "data/noisy"
    os.makedirs(outdir, exist_ok=True)

    base_seed = int(clean_path.split("seed")[-1].replace(".npz", ""))

    # Helper to save dataset with consistent format
    def save_dataset(filename, noisy_obs, noisy_actions):
        full_path = os.path.join(outdir, filename)

        save_dict = {
            "observations": noisy_obs,
            "actions": noisy_actions,
        }
        if returns is not None:
            save_dict["returns"] = returns
        if lengths is not None:
            save_dict["episode_lengths"] = lengths

        np.savez_compressed(full_path, **save_dict)
        print_progress(f"Saved noisy dataset → {full_path}")

    # ---------------------- FLIP ----------------------
    for p in flip_levels:
        noisy_actions = add_flip_noise(acts, p)
        fname = f"flip_p{p:.2f}_{N}_seed{seed}.npz"
        save_dataset(fname, obs, noisy_actions)

    # ------------------ GAUSSIAN ----------------------
    for s in gauss_levels:
        noisy_obs = add_gaussian_noise(obs, s)
        fname = f"gauss_s{s:.2f}_{N}_seed{seed}.npz"
        save_dataset(fname, noisy_obs, acts)

    # ------------------ DROPOUT -----------------------
    for p in dropout_levels:
        noisy_obs = add_dropout_noise(obs, p)
        fname = f"dropout_p{p:.2f}_{N}_seed{seed}.npz"
        save_dataset(fname, noisy_obs, acts)

    # ------------------ BIAS --------------------------
    for s in bias_levels:
        noisy_obs = add_bias_noise(obs, s)
        fname = f"bias_s{s:.2f}_{N}_seed{seed}.npz"
        save_dataset(fname, noisy_obs, acts)

    # ------------------ JITTER ------------------------
    for p in jitter_levels:
        noisy_obs = add_jitter_noise(obs, p)
        fname = f"jitter_p{p:.2f}_{N}_seed{seed}.npz"
        save_dataset(fname, noisy_obs, acts)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=str, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--flip_levels", nargs="+", type=float, default=[0.1, 0.45])
    parser.add_argument("--gauss_levels", nargs="+", type=float, default=[0.1, 0.45])
    parser.add_argument("--dropout_levels", nargs="+", type=float, default=[0.1, 0.45])
    parser.add_argument("--bias_levels", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--jitter_levels", nargs="+", type=float, default=[0.1, 0.45])

    args = parser.parse_args()

    generate_noisy_datasets(
        args.clean,
        args.N,
        args.seed,
        args.flip_levels,
        args.gauss_levels,
        args.dropout_levels,
        args.bias_levels,
        args.jitter_levels,
    )
