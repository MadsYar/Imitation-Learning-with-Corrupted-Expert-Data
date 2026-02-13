# Imitation Learning Under Noisy Expert Demonstrations

This project compares Behavioral Cloning (BC) and DAgger on CartPole when expert demonstrations are corrupted by different noise processes. It builds clean and noisy datasets, trains policies, evaluates them across seeds and produces plots that summarize performance.

## Goals

- Compare BC vs DAgger under controlled noise types and levels.
- Study how dataset size and filtering (episode-level, step-level) affect robustness.
- Provide a repeatable pipeline for dataset generation, training, evaluation, and plotting.

## Project Structure

- [collect_trajectories.py](collect_trajectories.py): Collects expert demonstrations into an NPZ dataset.
- [noisy_data.py](noisy_data.py): Generates noisy variants of expert data.
- [train_expert.py](train_expert.py): Trains a DQN expert on CartPole.
- [train_bc.py](train_bc.py): Trains a BC policy with optional episode filtering.
- [train_dagger.py](train_dagger.py): Trains a DAgger policy from an initial dataset.
- [evaluate.py](evaluate.py): Evaluates expert, BC, and DAgger models and appends results to CSV.
- [plotting.py](plotting.py): Aggregates results from CSV and generates plots.
- [run.py](run.py): Full end-to-end pipeline (train, collect, corrupt, train, evaluate, plot).
- [strict_*.py](strict_bc.py): Alternate strict filtering experiment scripts.
- [utils.py](utils.py): Shared helpers (seeding, device selection, directories, logging).

## Method Overview

1. **Train expert** using DQN on CartPole.
2. **Collect clean demonstrations** with the trained expert.
3. **Generate noisy datasets** by corrupting observations or actions.
4. **Train BC** on clean and noisy datasets, with optional filtering.
5. **Train DAgger** using the expert for iterative relabeling.
6. **Evaluate** all policies and record results.
7. **Plot** aggregate metrics and comparisons.

## Noise Types

- **Flip**: Randomly flips discrete actions.
- **Gaussian**: Adds Gaussian noise to observations.
- **Dropout**: Zeros random elements in observations.
- **Bias**: Adds a fixed bias vector per dataset.
- **Jitter**: Repeats previous observations with probability $p$.

## Filtering Strategies

- **Episode filtering (EPF)**: Keeps episodes with return above a threshold.
- **Step filtering**: Keeps steps where the expert agrees or from high-return episodes.

## Requirements

- Python 3.9+
- `gymnasium`
- `stable-baselines3`
- `torch`
- `numpy`
- `matplotlib`

Install dependencies:

```bash
pip install gymnasium stable-baselines3 torch numpy matplotlib
```

## Quickstart

Run the full experiment pipeline:

```bash
python run.py
```

This will:
- Train the expert
- Collect clean datasets
- Generate noisy datasets
- Train BC and DAgger models
- Evaluate all models
- Produce plots in the `plots/` directory

## Manual Usage Examples

Train an expert:

```bash
python train_expert.py --timesteps 300000 --seed 123 --out models/expert_final.zip
```

Collect clean demonstrations:

```bash
python collect_trajectories.py --expert models/expert_final.zip --episodes 500 --seed 123 --out data/expert/clean_500_seed123.npz
```

Generate noisy datasets:

```bash
python noisy_data.py --clean data/expert/clean_500_seed123.npz --N 500 --seed 123
```

Train BC on a dataset:

```bash
python train_bc.py --data data/expert/clean_500_seed123.npz --out models/bc_seed0_clean500.pt --epochs 20
```

Train DAgger on a dataset:

```bash
python train_dagger.py --expert models/expert_final.zip --data data/expert/clean_500_seed123.npz --out models/dagger_seed0_clean500.pt
```

Evaluate a model:

```bash
python evaluate.py --model models/bc_seed0_clean500.pt --episodes 100 --seed 123
```

Generate plots from evaluations:

```bash
python plotting.py
```

## Outputs

- `data/`: Clean and noisy datasets.
- `models/`: Trained BC and DAgger policies.
- `logs/evaluation_fixed.csv`: Evaluation table for plotting.
- `plots/`: Aggregated figures for comparisons.

## Results (Example)

These plots are generated after a full run and summarize the main comparisons. They are meant as quick visual checks rather than definitive benchmarks (results can shift slightly across seeds and hardware).

- Baseline clean performance: [plots/baselines/baseline_clean.png](plots/baselines/baseline_clean.png)
- Noise impact by type and dataset size: [plots/noise/gauss_size500.png](plots/noise/gauss_size500.png), [plots/noise/flip_size500.png](plots/noise/flip_size500.png)
- Filtering effects (clean and noisy): [plots/filtering/clean_filtering.png](plots/filtering/clean_filtering.png), [plots/filtering/gauss_filtering_size500.png](plots/filtering/gauss_filtering_size500.png)
- Aggregate stability across seeds: [plots/confidence/confidence.png](plots/confidence/confidence.png)

Interpretation note: a steeper performance drop under higher noise indicates reduced robustness. If DAgger (or filtering variants) degrades less than BC for the same noise level, it suggests better resilience to noisy demonstrations.

## Notes

- Default environment is `CartPole-v1`.
- The pipeline uses seeds for reproducibility.
- Running the full pipeline can take time depending on hardware.

