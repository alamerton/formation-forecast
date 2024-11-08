# %%
import numpy as np
from utils.geometric_mean_odds import get_geometric_mean

# %%
agi_forecasts = [
    [0.2809, 0.6433, 0.7649, 0.8673, 0.9250],  # Epoch model-based
    [0.6572, 0.9334, 0.9603, 0.9697, 0.9745],  # Metaculus weakly general
    [0.4183, 0.8478, 0.9002, 0.9279, 0.9552],  # Metaculus general
    [0.1000, 0.5324, 0.8027, 0.9999, 0.9999],  # AI Impacts
    [0.3100, 0.6480, 0.7380, 0.8280, 0.9180],  # Samotsvety
]

# %%
weights = np.ones(len(agi_forecasts)) / len(agi_forecasts)
print(weights)
# %%
agi_probs = []

for col in range(len(agi_forecasts[0])):
    column_values = [row[col] for row in agi_forecasts]
    print(column_values)
    prob = get_geometric_mean(column_values)
    print("Prob: ", prob)
    agi_probs.append(prob)
# %%
print(agi_probs)

# %%
