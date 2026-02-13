import torch
import torch.nn as nn
from strict_data import load_dataset

class BC(nn.Module):
    def __init__(self, obs_dim=4, act_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )

    def forward(self, x):
        return self.net(x)


def train_bc(dataset_path, filter_mode=None, epf_threshold=None, device="cpu"):
    obs, acts = load_dataset(dataset_path, filter_mode, epf_threshold)

    # Convert to float32 and reshape acts to match model output
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    acts = torch.tensor(acts, dtype=torch.float32).unsqueeze(-1).to(device)  # Add dimension

    model = BC().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(20):  # small number of epochs
        pred = model(obs)
        loss = loss_fn(pred, acts)
        optim.zero_grad()
        loss.backward()
        optim.step()

    return model