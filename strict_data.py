import numpy as np

def flatten_episode_list(arr):
    """
    arr is something like:
    [
        [obs1, obs2, obs3, ...],   # episode 1
        [obs1, obs2, obs3, ...],   # episode 2
        ...
    ]

    This converts into a flat array (N, 4)
    """
    out = []
    for episode in arr:
        for step in episode:
            out.append(step)
    return np.array(out)


def load_dataset(path, filter_mode=None, epf_threshold=None):
    data = np.load(path, allow_pickle=True)
    print("[DEBUG] Keys in dataset:", list(data.keys()))

    obs = data["observations"]
    acts = data["actions"]

    # Check if data is already flat or nested
    if obs.dtype == object:
        # Data is nested by episode, flatten it
        obs = flatten_episode_list(obs)
        acts = flatten_episode_list(acts)

    # EPISODE FILTERING
    if filter_mode == "episode":
        returns = data["returns"]
        episode_lengths = data["episode_lengths"]
        
        # Filter episodes by return threshold
        idx_keep = returns >= epf_threshold
        
        obs_kept = []
        acts_kept = []
        
        # Iterate through episodes using episode_lengths
        start_idx = 0
        for i, length in enumerate(episode_lengths):
            end_idx = start_idx + int(length)
            
            if idx_keep[i]:
                # Keep this episode
                obs_kept.append(obs[start_idx:end_idx])
                acts_kept.append(acts[start_idx:end_idx])
            
            start_idx = end_idx
        
        # Concatenate kept episodes
        obs = np.concatenate(obs_kept) if obs_kept else np.array([])
        acts = np.concatenate(acts_kept) if acts_kept else np.array([])

    return obs, acts
