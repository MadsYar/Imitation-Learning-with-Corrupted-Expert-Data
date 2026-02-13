# plotting.py
import os
import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

EVAL_CSV = "logs/evaluation_fixed.csv"


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_noise(dataset_key):
    """
    Extract noise type + noise level from dataset key.
    Examples:
        flip_p0.10_500 → ("flip", 0.10)
        gauss_s0.45_1000 → ("gauss", 0.45)
        clean500 → ("", None)
    """
    # flip_p0.10
    m = re.match(r"(flip)_p([0-9]*\.?[0-9]+)", dataset_key)
    if m:
        return m.group(1), float(m.group(2))

    # gauss_s0.10
    m = re.match(r"(gauss)_s([0-9]*\.?[0-9]+)", dataset_key)
    if m:
        return m.group(1), float(m.group(2))

    # dropout_p...
    m = re.match(r"(dropout)_p([0-9]*\.?[0-9]+)", dataset_key)
    if m:
        return m.group(1), float(m.group(2))

    # bias_s...
    m = re.match(r"(bias)_s([0-9]*\.?[0-9]+)", dataset_key)
    if m:
        return m.group(1), float(m.group(2))

    # jitter_p...
    m = re.match(r"(jitter)_p([0-9]*\.?[0-9]+)", dataset_key)
    if m:
        return m.group(1), float(m.group(2))

    return "", None

def parse_size(dataset_key):
    """
    Extract size (500, 1000) from dataset keys like:
      clean500
      clean500_step
      clean1000_epf
      gauss_s0.10_500
    """
    # first try clean datasets
    m = re.match(r"clean(\d+)", dataset_key)
    if m:
        return m.group(1)

    # then try noisy datasets
    m = re.search(r"_(\d+)", dataset_key)
    if m:
        return m.group(1)

    return "?"



def parse_variant(dataset_key):
    """
    clean500_step → "step"
    clean500_epf → "epf"
    clean500 → "full"
    """
    if dataset_key.endswith("_step"):
        return "step"
    if dataset_key.endswith("_epf"):
        return "epf"
    return "full"


# ---------------------------------------------------------------
# READ CSV
# ---------------------------------------------------------------

def read_evaluations(path=EVAL_CSV):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] No evaluation file at {p}")
        return []

    rows = []
    with p.open("r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if len(line) < 7:
                continue

            name, algo, dataset_key, mean_r, std_r, succ, seed = line
            mean_r = float(mean_r)
            std_r = float(std_r)
            succ = float(succ)
            seed = int(seed)

            noise_type, noise_level = parse_noise(dataset_key)
            size = parse_size(dataset_key)
            variant = parse_variant(dataset_key)

            rows.append({
                "name": name,
                "algo": algo,
                "dataset": dataset_key,
                "variant": variant,
                "mean": mean_r,
                "std": std_r,
                "success": succ,
                "seed": seed,
                "noise_type": noise_type,
                "noise_level": noise_level,
                "size": size,
            })

    print(f"[INFO] Loaded {len(rows)} evaluations")
    return rows


# ---------------------------------------------------------------
# AGGREGATION
# ---------------------------------------------------------------

def aggregate(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in keys)
        groups[key].append(r)

    results = []
    for key, items in groups.items():
        means = np.array([x["mean"] for x in items])
        stds = np.array([x["std"] for x in items])
        succs = np.array([x["success"] for x in items])

        entry = {"key": key}
        entry["mean"] = float(means.mean())
        entry["std_between_seeds"] = float(means.std())
        entry["success"] = float(succs.mean())
        entry["n_seeds"] = len(items)

        for k, v in zip(keys, key):
            entry[k] = v

        results.append(entry)

    return results


# ---------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------

def plot_baselines(rows):
    ensure_dir("plots/baselines")

    clean_rows = [r for r in rows if r["noise_type"] == ""]
    agg = aggregate(clean_rows, ["algo"])

    algos = ["expert", "bc", "dagger"]
    vals = [next((a["mean"] for a in agg if a["algo"] == algo), None) for algo in algos]

    plt.figure(figsize=(6, 4))
    plt.bar([a.upper() for a in algos], vals)
    plt.ylabel("Average Return")
    plt.title("Baseline Performance (Clean Data)")
    out = "plots/baselines/baseline_clean.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT] {out}")


def plot_noise_effects(rows):
    ensure_dir("plots/noise")

    noisy = [r for r in rows if r["noise_type"] != ""]
    agg = aggregate(noisy, ["algo", "noise_type", "noise_level", "size"])

    noise_types = sorted({a["noise_type"] for a in agg})

    for nt in noise_types:
        levels = sorted({a["noise_level"] for a in agg if a["noise_type"] == nt})
        sizes = sorted({a["size"] for a in agg})

        for S in sizes:
            subset = [a for a in agg if a["size"] == S and a["noise_type"] == nt]

            if not subset:
                continue

            plt.figure(figsize=(7, 4))
            xs = levels

            for algo in ["bc", "dagger"]:
                ys = [
                    next((x["mean"] for x in subset if x["algo"] == algo and x["noise_level"] == lvl), np.nan)
                    for lvl in xs
                ]
                yerr = [
                    next((x["std_between_seeds"] for x in subset if x["algo"] == algo and x["noise_level"] == lvl), 0.0)
                    for lvl in xs
                ]
                plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label=algo.upper())

            plt.xlabel("Noise Level")
            plt.ylabel("Average Return")
            plt.title(f"{nt} Noise — N={S}")
            plt.legend()
            plt.tight_layout()

            out = f"plots/noise/{nt}_size{S}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"[PLOT] {out}")


def plot_filtering_clean(rows):
    ensure_dir("plots/filtering")

    # Only BC on CLEAN datasets
    clean = [r for r in rows if r["noise_type"] == "" and r["algo"] == "bc"]

    # Aggregate by (size, variant)
    agg = aggregate(clean, ["size", "variant"])

    sizes = sorted({a["size"] for a in agg}, key=lambda x: int(x))

    labels = []
    full_vals, step_vals, epf_vals = [], [], []

    for S in sizes:
        labels.append(f"N={S}")

        full_vals.append(next((x["mean"] for x in agg if x["size"] == S and x["variant"] == "full"), np.nan))
        step_vals.append(next((x["mean"] for x in agg if x["size"] == S and x["variant"] == "step"), np.nan))
        epf_vals.append(next((x["mean"] for x in agg if x["size"] == S and x["variant"] == "epf"), np.nan))

    x = np.arange(len(sizes))
    w = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, full_vals, w, label="Full")
    plt.bar(x,     step_vals, w, label="Step-filtered")
    plt.bar(x + w, epf_vals,  w, label="EPF")

    plt.xticks(x, labels)
    plt.ylabel("Average Return")
    plt.title("Filtering (Clean Data)")
    plt.legend()
    plt.tight_layout()

    out = "plots/filtering/clean_filtering.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"[PLOT] {out}")


def plot_filtering_noisy(rows):
    """
    Filtering effects on NOISY datasets:
    For each noise type + dataset size, compare
    Full vs Step-filtered vs EPF across noise levels.
    """
    ensure_dir("plots/filtering")

    noisy = [r for r in rows if r["noise_type"] != "" and r["algo"] == "bc"]
    if not noisy:
        print("[PLOT] No noisy rows for filtering plot.")
        return

    agg = aggregate(noisy, ["noise_type", "noise_level", "size", "variant"])

    noise_types = sorted({a["noise_type"] for a in agg})
    for nt in noise_types:
        sizes = sorted({a["size"] for a in agg if a["noise_type"] == nt})
        for S in sizes:
            subset = [a for a in agg if a["noise_type"] == nt and a["size"] == S]
            if not subset:
                continue

            levels = sorted({a["noise_level"] for a in subset})
            xs = levels

            plt.figure(figsize=(7, 4))

            for variant, label in [("full", "Full"),
                                   ("step", "Step-filtered"),
                                   ("epf", "EPF")]:
                ys = [
                    next((x["mean"] for x in subset
                          if x["variant"] == variant and x["noise_level"] == lvl),
                         np.nan)
                    for lvl in xs
                ]
                yerr = [
                    next((x["std_between_seeds"] for x in subset
                          if x["variant"] == variant and x["noise_level"] == lvl),
                         0.0)
                    for lvl in xs
                ]

                # skip variant if it doesn't exist at all
                if all(np.isnan(ys)):
                    continue

                plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, label=label)

            plt.xlabel("Noise Level")
            plt.ylabel("Average Return")
            plt.title(f"{nt} Noise — Filtering — N={S}")
            plt.legend()
            plt.tight_layout()

            out = f"plots/filtering/{nt}_filtering_size{S}.png"
            plt.savefig(out, dpi=150)
            plt.close()
            print(f"[PLOT] {out}")


def plot_confidence(rows):
    """
    Aggregated, readable confidence plot.
    Groups models into:
        - Expert
        - BC Clean
        - BC Clean Step
        - BC Clean EPF
        - BC Noisy
        - BC Noisy Step
        - BC Noisy EPF
        - DAgger Clean
        - DAgger Noisy

    Produces a clean bar plot with one bar per category.
    """

    ensure_dir("plots/confidence")

    # 1. Build high-level categories
    categories = {
        "EXPERT": [],
        "BC Clean (Full)": [],
        "BC Clean (Step)": [],
        "BC Clean (EPF)": [],
        "BC Noisy (Full)": [],
        "BC Noisy (Step)": [],
        "BC Noisy (EPF)": [],
        "DAgger Clean": [],
        "DAgger Noisy": [],
    }

    for r in rows:
        algo = r["algo"]
        nt = r["noise_type"]
        var = r["variant"]

        # expert
        if algo == "expert":
            categories["EXPERT"].append(r)
            continue

        # BC vs DAgger
        if algo == "bc":
            if nt == "":   # clean
                if var == "full":
                    categories["BC Clean (Full)"].append(r)
                elif var == "step":
                    categories["BC Clean (Step)"].append(r)
                elif var == "epf":
                    categories["BC Clean (EPF)"].append(r)
            else:          # noisy
                if var == "full":
                    categories["BC Noisy (Full)"].append(r)
                elif var == "step":
                    categories["BC Noisy (Step)"].append(r)
                elif var == "epf":
                    categories["BC Noisy (EPF)"].append(r)

        elif algo == "dagger":
            if nt == "":
                categories["DAgger Clean"].append(r)
            else:
                categories["DAgger Noisy"].append(r)

    # 2. Compute stats
    labels = []
    means = []
    errs = []

    for name, items in categories.items():
        if len(items) == 0:
            continue
        arr = np.array([x["mean"] for x in items])
        labels.append(name)
        means.append(arr.mean())
        errs.append(arr.std())

    # 3. Plot
    x = np.arange(len(labels))

    plt.figure(figsize=(12, 6))
    plt.bar(x, means, yerr=errs, capsize=5)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Average Return")
    plt.title("Performance Variance Across Seeds (Aggregated)")
    plt.tight_layout()

    out = "plots/confidence/confidence.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[PLOT] {out}")



# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    rows = read_evaluations()
    if not rows:
        print("[ERROR] No rows loaded — cannot plot.")
        return

    plot_baselines(rows)
    plot_noise_effects(rows)
    plot_filtering_clean(rows)
    plot_filtering_noisy(rows)
    plot_confidence(rows)

    print("[DONE] All plots generated.")


if __name__ == "__main__":
    main()
