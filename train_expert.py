# train_expert.py
import argparse
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from utils import make_dirs, set_seed, print_progress


# -------------------------------------------------------------------
# TRAIN EXPERT
# -------------------------------------------------------------------
def train_expert(
    timesteps: int = 300_000,
    seed: int = 123,
    out: str = "models/expert_final.zip",
):
    make_dirs()
    set_seed(seed)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    print_progress(f"Training expert policy for {timesteps} steps...")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50_000,
        batch_size=64,
        learning_starts=1000,
        train_freq=4,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        gamma=0.99,
        tau=1.0,
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path="models/",
        log_path="logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)

    print_progress(f"Saving expert model → {out}")
    model.save(out)

    env.close()
    eval_env.close()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", type=str, default="models/expert_final.zip")
    args = parser.parse_args()

    train_expert(
        timesteps=args.timesteps,
        seed=args.seed,
        out=args.out,
    )
