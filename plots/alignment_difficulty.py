# %%
from utils.geometric_mean_odds import get_geometric_mean

# %%
values = [
    0.4,
    0.15,
    0.75,
    0.65,
    0.75,
    0.7,
    0.95,
    0.5,
    0.001,
    0.3,
    0.8,
]
geometric_mean_value = get_geometric_mean(values)
print(f"The geometric mean of the odds is: {geometric_mean_value:.4f}")
print(f"Making the probability: {1/geometric_mean_value}")

# %%
# Histogram
