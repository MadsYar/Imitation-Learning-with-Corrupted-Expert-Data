# import matplotlib.pyplot as plt
# from strict_bc import train_bc
# from strict_evaluation import evaluate_policy
# import torch
# import os
# import glob

# # ---------------------------------------------------
# # CONFIG
# # ---------------------------------------------------
# STRICT_THRESHOLD = 400     # episodes with return < 400 are discarded
# DEVICE = "mps"             # or "cpu", "cuda"

# # All noise types to process
# NOISE_TYPES = ["gauss", "flip", "bias", "dropout", "jitter"]

# # ---------------------------------------------------
# # ENV CONSTRUCTOR
# # ---------------------------------------------------
# def make_env():
#     import gymnasium as gym
#     return gym.make("CartPole-v1")


# # ---------------------------------------------------
# # HELPER FUNCTIONS
# # ---------------------------------------------------
# def extract_noise_level(path):
#     """
#     Extracts actual noise number from filenames such as:
#     gauss_s0.10_500_seed123.npz → 0.10
#     flip_p0.45_500_seed123.npz  → 0.45
#     bias_s0.05_500_seed123.npz  → 0.05
#     """
#     name = os.path.basename(path)
#     parts = name.split("_")
    
#     # Second part contains the noise level (s0.10, p0.45, etc.)
#     code = parts[1]  # 's0.10' or 'p0.45'
#     val = code[1:]   # remove leading character → "0.10"
#     return float(val)


# def get_noise_datasets(noise_type, data_dir="data/noisy"):
#     """
#     Find all datasets for a given noise type.
#     Returns a sorted list of paths.
#     """
#     pattern = os.path.join(data_dir, f"{noise_type}_*.npz")
#     paths = glob.glob(pattern)
    
#     # Filter out _step.npz and _backup.npz files
#     paths = [p for p in paths if "_step" not in p and "_backup" not in p]
    
#     # Sort by noise level
#     paths.sort(key=extract_noise_level)
    
#     return paths


# # ---------------------------------------------------
# # PLOTTING FUNCTION
# # ---------------------------------------------------
# def plot_results(noise_type, paths, res):
#     noise_levels = [extract_noise_level(p) for p in paths]

#     full_mean = [m for (m, s) in res["full"]]
#     full_std  = [s for (m, s) in res["full"]]
#     epf_mean  = [m for (m, s) in res["epf"]]
#     epf_std   = [s for (m, s) in res["epf"]]

#     plt.figure(figsize=(8, 5))
#     plt.errorbar(noise_levels, full_mean, yerr=full_std, label="Full", marker="o")
#     plt.errorbar(
#         noise_levels, epf_mean, yerr=epf_std,
#         label=f"EPF ≥ {STRICT_THRESHOLD}", marker="o"
#     )

#     plt.title(f"{noise_type.capitalize()} Noise – Strict Episode Filtering")
#     plt.xlabel("Noise Level")
#     plt.ylabel("Average Return")
#     plt.grid(True)
#     plt.legend()

#     save_path = f"plots/strict_filter_{noise_type}.png"
#     os.makedirs("plots", exist_ok=True)
#     plt.savefig(save_path)
#     plt.close()

#     print(f"[SAVED] {save_path}")


# # ---------------------------------------------------
# # MAIN EXPERIMENT
# # ---------------------------------------------------
# def run():
#     results = {}

#     for noise_type in NOISE_TYPES:
#         print(f"\n{'='*60}")
#         print(f"Processing noise type: {noise_type.upper()}")
#         print('='*60)
        
#         # Find all datasets for this noise type
#         paths = get_noise_datasets(noise_type)
        
#         if not paths:
#             print(f"[WARNING] No datasets found for noise type: {noise_type}")
#             continue
        
#         print(f"Found {len(paths)} datasets:")
#         for p in paths:
#             print(f"  - {os.path.basename(p)}")
        
#         results[noise_type] = {"full": [], "epf": []}

#         for path in paths:
#             print(f"\n{'='*60}")
#             print(f"Training on dataset: {os.path.basename(path)}")
#             print('='*60)

#             try:
#                 # -------- FULL (NO FILTERING) --------
#                 model_full = train_bc(path, filter_mode=None, device=DEVICE)
#                 mean_full, std_full = evaluate_policy(model_full, make_env, device=DEVICE)
#                 results[noise_type]["full"].append((mean_full, std_full))
#                 print(f"Full dataset: {mean_full:.2f} ± {std_full:.2f}")

#                 # -------- STRICT EPF FILTERING (≥ threshold) --------
#                 model_epf = train_bc(
#                     path,
#                     filter_mode="episode",
#                     epf_threshold=STRICT_THRESHOLD,
#                     device=DEVICE
#                 )
#                 mean_epf, std_epf = evaluate_policy(model_epf, make_env, device=DEVICE)
#                 results[noise_type]["epf"].append((mean_epf, std_epf))
#                 print(f"EPF filtered: {mean_epf:.2f} ± {std_epf:.2f}")
                
#             except Exception as e:
#                 print(f"[ERROR] Failed to process {path}: {e}")
#                 continue

#         # Produce a PNG for each noise type (if we have results)
#         if results[noise_type]["full"]:
#             plot_results(noise_type, paths, results[noise_type])
#         else:
#             print(f"[WARNING] No successful results for {noise_type}, skipping plot")

#     print("\n\n" + "="*60)
#     print("=== EXPERIMENT COMPLETE ===")
#     print("="*60)
    
#     # Print summary
#     for noise_type, res in results.items():
#         if res["full"]:
#             print(f"\n{noise_type.upper()}:")
#             print(f"  Full datasets processed: {len(res['full'])}")
#             print(f"  EPF datasets processed: {len(res['epf'])}")


# # ---------------------------------------------------
# if __name__ == "__main__":
#     run()



import matplotlib.pyplot as plt
from strict_bc import train_bc
from strict_evaluation import evaluate_policy
import torch
import os
import numpy as np
import glob

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
STRICT_THRESHOLD = 400     # episodes with return < 400 are discarded
DEVICE = "mps"             # or "cpu", "cuda"

# All noise types to process
NOISE_TYPES = ["gauss", "flip", "bias", "dropout", "jitter"]

# ---------------------------------------------------
# ENV CONSTRUCTOR
# ---------------------------------------------------
def make_env():
    import gymnasium as gym
    return gym.make("CartPole-v1")


# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def extract_noise_level(path):
    """
    Extracts actual noise number from filenames such as:
    gauss_s0.10_500_seed123.npz → 0.10
    flip_p0.45_500_seed123.npz  → 0.45
    bias_s0.05_500_seed123.npz  → 0.05
    """
    name = os.path.basename(path)
    parts = name.split("_")
    
    # Second part contains the noise level (s0.10, p0.45, etc.)
    code = parts[1]  # 's0.10' or 'p0.45'
    val = code[1:]   # remove leading character → "0.10"
    return float(val)


def get_noise_datasets(noise_type, data_dir="data/noisy"):
    """
    Find all datasets for a given noise type.
    Returns a sorted list of paths.
    """
    pattern = os.path.join(data_dir, f"{noise_type}_*.npz")
    paths = glob.glob(pattern)
    
    # Filter out _step.npz and _backup.npz files
    paths = [p for p in paths if "_step" not in p and "_backup" not in p]
    
    # Sort by noise level
    paths.sort(key=extract_noise_level)
    
    return paths


# ---------------------------------------------------
# PLOTTING FUNCTION
# ---------------------------------------------------
def plot_results(noise_type, paths, res):
    from collections import defaultdict
    
    # Group results by noise level
    noise_groups = defaultdict(lambda: {"full_means": [], "full_stds": [], "epf_means": [], "epf_stds": []})
    
    for i, path in enumerate(paths):
        noise_level = extract_noise_level(path)
        full_m, full_s = res["full"][i]
        epf_m, epf_s = res["epf"][i]
        
        noise_groups[noise_level]["full_means"].append(full_m)
        noise_groups[noise_level]["full_stds"].append(full_s)
        noise_groups[noise_level]["epf_means"].append(epf_m)
        noise_groups[noise_level]["epf_stds"].append(epf_s)
    
    # Average results for each noise level
    noise_levels = sorted(noise_groups.keys())
    full_means = [np.mean(noise_groups[nl]["full_means"]) for nl in noise_levels]
    full_stds = [np.mean(noise_groups[nl]["full_stds"]) for nl in noise_levels]
    epf_means = [np.mean(noise_groups[nl]["epf_means"]) for nl in noise_levels]
    epf_stds = [np.mean(noise_groups[nl]["epf_stds"]) for nl in noise_levels]

    plt.figure(figsize=(8, 5))
    plt.errorbar(noise_levels, full_means, yerr=full_stds, label="Full", marker="o", markersize=8)
    plt.errorbar(
        noise_levels, epf_means, yerr=epf_stds,
        label=f"EPF ≥ {STRICT_THRESHOLD}", marker="o", markersize=8
    )

    plt.title(f"{noise_type.capitalize()} Noise – Strict Episode Filtering")
    plt.xlabel("Noise Level")
    plt.ylabel("Average Return")
    plt.grid(True)
    plt.legend()

    save_path = f"plots/strict_filter_{noise_type}.png"
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"[SAVED] {save_path}")


# ---------------------------------------------------
# MAIN EXPERIMENT
# ---------------------------------------------------
def run():
    results = {}

    for noise_type in NOISE_TYPES:
        print(f"\n{'='*60}")
        print(f"Processing noise type: {noise_type.upper()}")
        print('='*60)
        
        # Find all datasets for this noise type
        paths = get_noise_datasets(noise_type)
        
        if not paths:
            print(f"[WARNING] No datasets found for noise type: {noise_type}")
            continue
        
        print(f"Found {len(paths)} datasets:")
        for p in paths:
            print(f"  - {os.path.basename(p)}")
        
        results[noise_type] = {"full": [], "epf": []}

        for path in paths:
            print(f"\n{'='*60}")
            print(f"Training on dataset: {os.path.basename(path)}")
            print('='*60)

            try:
                # -------- FULL (NO FILTERING) --------
                model_full = train_bc(path, filter_mode=None, device=DEVICE)
                mean_full, std_full = evaluate_policy(model_full, make_env, device=DEVICE)
                results[noise_type]["full"].append((mean_full, std_full))
                print(f"Full dataset: {mean_full:.2f} ± {std_full:.2f}")

                # -------- STRICT EPF FILTERING (≥ threshold) --------
                model_epf = train_bc(
                    path,
                    filter_mode="episode",
                    epf_threshold=STRICT_THRESHOLD,
                    device=DEVICE
                )
                mean_epf, std_epf = evaluate_policy(model_epf, make_env, device=DEVICE)
                results[noise_type]["epf"].append((mean_epf, std_epf))
                print(f"EPF filtered: {mean_epf:.2f} ± {std_epf:.2f}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process {path}: {e}")
                continue

        # Produce a PNG for each noise type (if we have results)
        if results[noise_type]["full"]:
            plot_results(noise_type, paths, results[noise_type])
        else:
            print(f"[WARNING] No successful results for {noise_type}, skipping plot")

    print("\n\n" + "="*60)
    print("=== EXPERIMENT COMPLETE ===")
    print("="*60)
    
    # Print summary
    for noise_type, res in results.items():
        if res["full"]:
            print(f"\n{noise_type.upper()}:")
            print(f"  Full datasets processed: {len(res['full'])}")
            print(f"  EPF datasets processed: {len(res['epf'])}")


# ---------------------------------------------------
if __name__ == "__main__":
    run()