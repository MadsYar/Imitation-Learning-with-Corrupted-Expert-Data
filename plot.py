import numpy as np
import matplotlib.pyplot as plt

data = np.load("logs/evaluations.npz")

noise_levels = data["noise_levels"]           # shape: [num_noise_levels]
mean_returns = data["mean_returns"]           # shape: [num_algorithms, num_noise_levels]
algorithms   = data["algorithms"]             # list of names

plt.figure(figsize=(7,4))

for i, alg in enumerate(algorithms):
    plt.plot(noise_levels, mean_returns[i], marker="o", label=alg)

plt.xlabel("Noise Level")
plt.ylabel("Mean Return")
plt.title("Mean Return vs Noise Level")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
