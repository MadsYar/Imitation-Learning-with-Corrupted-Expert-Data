import numpy as np
import torch

def evaluate_policy(model, make_env_fn, episodes=5, device="cpu"):
    returns = []
    env = make_env_fn()

    for _ in range(episodes):
        obs_reset = env.reset()
        
        # Handle both old gym and new gymnasium API
        if isinstance(obs_reset, tuple):
            obs = obs_reset[0]
        else:
            obs = obs_reset
            
        done = False
        total = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            act = model(obs_t).cpu().detach().numpy()
            
            # Flatten action if it's 2D (e.g., [[0.5]] -> [0.5])
            if act.ndim > 1:
                act = act.flatten()
            
            # CartPole expects a discrete action (0 or 1)
            # Convert continuous output to discrete
            act_discrete = int(act[0] > 0)
            
            step_result = env.step(act_discrete)
            
            # Handle both old gym and new gymnasium API for step
            if len(step_result) == 4:
                obs, r, done, _ = step_result
            else:  # len == 5 (new API with terminated, truncated)
                obs, r, terminated, truncated, _ = step_result
                done = terminated or truncated
                
            total += r

        returns.append(total)

    return np.mean(returns), np.std(returns)