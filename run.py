# # ================================
# # run.py — FULL EXPERIMENT VERSION
# # Matching debug logic but with full-scale settings
# # ================================
# import os
# import glob
# import numpy as np
# import gymnasium as gym
# from stable_baselines3 import DQN

# from utils import make_dirs, set_seed, print_progress
# from train_expert import train_expert
# from collect_trajectories import collect_trajectories
# from noisy_data import generate_noisy_datasets
# from train_bc import train_bc, get_dataset_key as bc_key
# from train_dagger import train_dagger, get_dataset_key as dagger_key
# from evaluate import evaluate_model
# import plotting


# # ================================================================
# # STEP-FILTERED DATASET (WORKING VERSION)
# # ================================================================
# def build_step_filtered_dataset(
#     expert_path: str,
#     clean_path: str,
#     out_path: str,
#     return_threshold: float = 200.0,
# ):
#     """Filters individual steps based on expert agreement or high-return episodes."""

#     if os.path.exists(out_path):
#         print_progress(f"[STEP] Already exists → {out_path}")
#         return

#     print_progress(f"[STEP] Building step-filtered dataset: {clean_path}")

#     data = np.load(clean_path)
#     obs = data["observations"]
#     acts = data["actions"]
#     returns = data["returns"]
#     lengths = data["episode_lengths"]

#     env = gym.make("CartPole-v1")
#     expert = DQN.load(expert_path, env=env, device="cpu")

#     f_obs, f_acts = [], []

#     idx = 0
#     for ret, L in zip(returns, lengths):
#         ep_obs = obs[idx:idx + L]
#         ep_acts = acts[idx:idx + L]

#         if ret >= return_threshold:
#             f_obs.extend(ep_obs)
#             f_acts.extend(ep_acts)
#         else:
#             for o, a in zip(ep_obs, ep_acts):
#                 exp_a, _ = expert.predict(o, deterministic=True)
#                 if int(exp_a) == int(a):
#                     f_obs.append(o)
#                     f_acts.append(a)

#         idx += L

#     np.savez_compressed(
#         out_path,
#         observations=np.array(f_obs, dtype=np.float32),
#         actions=np.array(f_acts, dtype=np.int64),
#     )

#     print_progress(f"[STEP] Saved step-filtered → {out_path}")
#     env.close()


# # ================================================================
# # MAIN FULL PIPELINE
# # ================================================================
# def main():
#     make_dirs()

#     # ----------------
#     # CONFIG
#     # ----------------
#     SEEDS = [0, 1, 2]
#     EPISODES_CLEAN = [500, 1000]

#     NOISE_LEVELS = {
#         "flip":   [0.10, 0.45],
#         "gauss":  [0.10, 0.45],
#         "dropout": [0.10, 0.45],
#         "bias":   [0.05, 0.20],
#         "jitter": [0.10, 0.45],
#     }

#     RETURN_THRESHOLD = 200

#     # remove old evaluation file
#     csv_path = "logs/evaluation_fixed.csv"
#     if os.path.exists(csv_path):
#         os.remove(csv_path)
#         print_progress("Deleted old evaluation_fixed.csv")

#     # -------------------------
#     # 1) TRAIN EXPERT
#     # -------------------------
#     expert_path = "models/expert_final.zip"
#     print_progress("=== TRAINING EXPERT ===")
#     train_expert(
#         timesteps=300_000,
#         seed=123,
#         out=expert_path
#     )

#     # -------------------------
#     # 2) COLLECT CLEAN DEMOS
#     # -------------------------
#     print_progress("=== COLLECTING CLEAN DATASETS ===")
#     clean_paths = []

#     for N in EPISODES_CLEAN:
#         out = f"data/expert/clean_{N}_seed123.npz"
#         collect_trajectories(
#             expert_path=expert_path,
#             episodes=N,
#             seed=123,
#             out_path=out
#         )
#         clean_paths.append(out)

#     # -------------------------
#     # 3) GENERATE NOISY DATASETS
#     # -------------------------
#     print_progress("=== GENERATING NOISY DATASETS ===")

#     for clean_path in clean_paths:
#         N = int(clean_path.split("_")[1])
#         generate_noisy_datasets(
#             clean_path,
#             N,
#             123,                            # seed
#             NOISE_LEVELS["flip"],
#             NOISE_LEVELS["gauss"],
#             NOISE_LEVELS["dropout"],
#             NOISE_LEVELS["bias"],
#             NOISE_LEVELS["jitter"],
#         )

#     noisy_paths = sorted(glob.glob("data/noisy/*.npz"))

#     # -------------------------
#     # 4) STEP FILTER CLEAN DATA
#     # -------------------------
#     print_progress("=== STEP FILTERING CLEAN DATA ===")
#     step_paths = []

#     for clean_path in clean_paths:
#         base = os.path.basename(clean_path).replace(".npz", "")
#         out = f"data/expert/{base}_step.npz"

#         build_step_filtered_dataset(
#             expert_path=expert_path,
#             clean_path=clean_path,
#             out_path=out,
#             return_threshold=RETURN_THRESHOLD
#         )

#         step_paths.append(out)

#     # -------------------------
#     # 5) TRAIN (BC + DAGGER)
#     # -------------------------
#     for seed in SEEDS:
#         print_progress(f"=== TRAINING SEED {seed} ===")
#         set_seed(seed)

#         # BC full (clean)
#         for path in clean_paths:
#             key = bc_key(path)
#             out = f"models/bc_seed{seed}_{key}.pt"
#             train_bc(path, out, epochs=10, seed=seed)

#         # BC episode filtered
#         for path in clean_paths:
#             key = bc_key(path) + "_epf"
#             out = f"models/bc_seed{seed}_{key}.pt"
#             train_bc(path, out, epochs=10, seed=seed,
#                      filter_bad=True, return_threshold=RETURN_THRESHOLD)

#         # BC step filtered
#         for path in step_paths:
#             key = bc_key(path)
#             out = f"models/bc_seed{seed}_{key}.pt"
#             train_bc(path, out, epochs=10, seed=seed)

#         # BC noisy
#         for path in noisy_paths:
#             key = bc_key(path)
#             out = f"models/bc_seed{seed}_{key}.pt"
#             train_bc(path, out, epochs=10, seed=seed)

#         # DAgger (clean + noisy)
#         dagger_paths = clean_paths + noisy_paths
#         for path in dagger_paths:
#             key = dagger_key(path)
#             out = f"models/dagger_seed{seed}_{key}.pt"
#             train_dagger(
#                 expert_path=expert_path,
#                 init_dataset_path=path,
#                 out_model_path=out,
#                 iterations=3,
#                 episodes_per_iter=5,
#                 epochs_per_iter=3,
#                 batch_size=64,
#                 lr=1e-3,
#                 seed=seed,
#             )

#         # -------------------------
#         # 6) EVALUATION
#         # -------------------------
#         print_progress("=== EVALUATING MODELS ===")

#         evaluate_model(expert_path, episodes=100, seed=seed)

#         for m in sorted(glob.glob("models/*.pt")):
#             if f"seed{seed}_" in m:
#                 evaluate_model(m, episodes=100, seed=seed)

#     # -------------------------
#     # 7) PLOTTING
#     # -------------------------
#     print_progress("=== GENERATING PLOTS ===")
#     plotting.main()

#     print_progress("=== FULL PIPELINE COMPLETE ===")


# if __name__ == "__main__":
#     main()

# ================================
# run.py — FULL EXPERIMENT VERSION
# ================================
import os
import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN

from utils import make_dirs, set_seed, print_progress
from train_expert import train_expert
from collect_trajectories import collect_trajectories
from noisy_data import generate_noisy_datasets
from train_bc import train_bc, get_dataset_key as bc_key
from train_dagger import train_dagger, get_dataset_key as dagger_key
from evaluate import evaluate_model
import plotting


# ================================================================
# STEP-FILTERED DATASET (generic: works for clean AND noisy)
# ================================================================
def build_step_filtered_dataset(
    expert_path: str,
    data_path: str,
    out_path: str,
    return_threshold: float = 200.0,
):
    """Filter individual steps based on expert agreement or high-return episodes."""

    if os.path.exists(out_path):
        print_progress(f"[STEP] Already exists → {out_path}")
        return

    print_progress(f"[STEP] Building step-filtered dataset: {data_path}")

    data = np.load(data_path)
    obs = data["observations"]
    acts = data["actions"]
    returns = data["returns"]
    lengths = data["episode_lengths"]

    env = gym.make("CartPole-v1")
    expert = DQN.load(expert_path, env=env, device="cpu")

    f_obs, f_acts = [], []

    idx = 0
    for ret, L in zip(returns, lengths):
        ep_obs = obs[idx:idx + L]
        ep_acts = acts[idx:idx + L]

        # keep whole episode if return is high
        if ret >= return_threshold:
            f_obs.extend(ep_obs)
            f_acts.extend(ep_acts)
        else:
            # otherwise, keep only steps where expert agrees
            for o, a in zip(ep_obs, ep_acts):
                exp_a, _ = expert.predict(o, deterministic=True)
                if int(exp_a) == int(a):
                    f_obs.append(o)
                    f_acts.append(a)

        idx += L

    np.savez_compressed(
        out_path,
        observations=np.array(f_obs, dtype=np.float32),
        actions=np.array(f_acts, dtype=np.int64),
    )

    print_progress(f"[STEP] Saved step-filtered → {out_path}")
    env.close()


# ================================================================
# MAIN FULL PIPELINE
# ================================================================
def main():
    make_dirs()

    SEEDS = [0, 1, 2]
    EPISODES_CLEAN = [500, 1000]

    NOISE_LEVELS = {
        "flip":   [0.10, 0.45],
        "gauss":  [0.10, 0.45],
        "dropout": [0.10, 0.45],
        "bias":   [0.05, 0.20],
        "jitter": [0.10, 0.45],
    }

    RETURN_THRESHOLD = 200

    # remove old evaluation file
    csv_path = "logs/evaluation_fixed.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
        print_progress("Deleted old evaluation_fixed.csv")

    # -------------------------
    # 1) TRAIN EXPERT  (only once; comment out after you have expert_final.zip)
    # -------------------------
    expert_path = "models/expert_final.zip"
    print_progress("=== TRAINING EXPERT ===")
    train_expert(
        timesteps=300_000,
        seed=123,
        out=expert_path
    )

    # -------------------------
    # 2) COLLECT CLEAN DEMOS  (also only once)
    # -------------------------
    print_progress("=== COLLECTING CLEAN DATASETS ===")
    clean_paths = []

    for N in EPISODES_CLEAN:
        out = f"data/expert/clean_{N}_seed123.npz"
        collect_trajectories(
            expert_path=expert_path,
            episodes=N,
            seed=123,
            out_path=out
        )
        clean_paths.append(out)

    # -------------------------
    # 3) GENERATE NOISY DATASETS  (only once)
    # -------------------------
    print_progress("=== GENERATING NOISY DATASETS ===")

    for clean_path in clean_paths:
        N = int(clean_path.split("_")[1])
        generate_noisy_datasets(
            clean_path,
            N,
            123,                            # seed
            NOISE_LEVELS["flip"],
            NOISE_LEVELS["gauss"],
            NOISE_LEVELS["dropout"],
            NOISE_LEVELS["bias"],
            NOISE_LEVELS["jitter"],
        )

    noisy_paths = sorted(glob.glob("data/noisy/*.npz"))

    # -------------------------
    # 4) STEP FILTER CLEAN + NOISY DATA  ✅ NEW
    # -------------------------
    print_progress("=== STEP FILTERING DATA (CLEAN + NOISY) ===")

    step_paths_clean = []
    for clean_path in clean_paths:
        base = os.path.basename(clean_path).replace(".npz", "")
        out = f"data/expert/{base}_step.npz"

        build_step_filtered_dataset(
            expert_path=expert_path,
            data_path=clean_path,
            out_path=out,
            return_threshold=RETURN_THRESHOLD
        )
        step_paths_clean.append(out)

    step_paths_noisy = []
    for noisy_path in noisy_paths:
        base = os.path.basename(noisy_path).replace(".npz", "")
        out = f"data/noisy/{base}_step.npz"

        build_step_filtered_dataset(
            expert_path=expert_path,
            data_path=noisy_path,
            out_path=out,
            return_threshold=RETURN_THRESHOLD
        )
        step_paths_noisy.append(out)

    # -------------------------
    # 5) TRAIN (BC + DAGGER)
    # -------------------------
    for seed in SEEDS:
        print_progress(f"=== TRAINING SEED {seed} ===")
        set_seed(seed)

        # -------- BC full (clean) --------
        for path in clean_paths:
            key = bc_key(path)
            out = f"models/bc_seed{seed}_{key}.pt"
            train_bc(path, out, epochs=10, seed=seed)

        # -------- BC episode filtered (clean) --------
        for path in clean_paths:
            key = bc_key(path) + "_epf"
            out = f"models/bc_seed{seed}_{key}.pt"
            train_bc(
                path, out, epochs=10, seed=seed,
                filter_bad=True, return_threshold=RETURN_THRESHOLD
            )

        # -------- BC step filtered (clean) --------
        for path in step_paths_clean:
            key = bc_key(path)
            out = f"models/bc_seed{seed}_{key}.pt"
            train_bc(path, out, epochs=10, seed=seed)

        # -------- BC full (noisy) --------
        for path in noisy_paths:
            key = bc_key(path)
            out = f"models/bc_seed{seed}_{key}.pt"
            train_bc(path, out, epochs=10, seed=seed)

        # -------- BC episode filtered (noisy) ✅ NEW --------
        for path in noisy_paths:
            key = bc_key(path) + "_epf"
            out = f"models/bc_seed{seed}_{key}.pt"
            train_bc(
                path, out, epochs=10, seed=seed,
                filter_bad=True, return_threshold=RETURN_THRESHOLD
            )

        # -------- BC step filtered (noisy) ✅ NEW --------
        for path in step_paths_noisy:
            key = bc_key(path)
            out = f"models/bc_seed{seed}_{key}.pt"
            train_bc(path, out, epochs=10, seed=seed)

        # -------- DAgger (clean + noisy, full only) --------
        dagger_paths = clean_paths + noisy_paths
        for path in dagger_paths:
            key = dagger_key(path)
            out = f"models/dagger_seed{seed}_{key}.pt"
            train_dagger(
                expert_path=expert_path,
                init_dataset_path=path,
                out_model_path=out,
                iterations=3,
                episodes_per_iter=5,
                epochs_per_iter=3,
                batch_size=64,
                lr=1e-3,
                seed=seed,
            )

        # -------------------------
        # 6) EVALUATION
        # -------------------------
        print_progress("=== EVALUATING MODELS ===")

        evaluate_model(expert_path, episodes=100, seed=seed)

        for m in sorted(glob.glob("models/*.pt")):
            if f"seed{seed}_" in m:
                evaluate_model(m, episodes=100, seed=seed)

    # -------------------------
    # 7) PLOTTING
    # -------------------------
    print_progress("=== GENERATING PLOTS ===")
    plotting.main()

    print_progress("=== FULL PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
