# %%
from utils.geometric_mean_odds import get_geometric_mean, odds_to_probability
from utils.logistic_interpolation import get_logistic_interpolation

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
geometric_mean_value = odds_to_probability(get_geometric_mean(values))
print(f"The geometric mean of the odds is: {geometric_mean_value:.4f}")

# %%
years = [2030, 2055, 2080, 2105, 2130]
k = 0.0161  # Growth rate
original_year = 2070
original_prob = geometric_mean_value
alignment_distribution = get_logistic_interpolation(
    years, original_year, original_prob, k
)
# %%
print("\nProbability estimates for alignment difficulty:")
print(alignment_distribution.to_string(index=False))

# Verify we hit close to our target probability
target_year = get_logistic_interpolation(
    [original_year], original_year, original_prob, k
)
print(
    f"\nProbability for {original_year} (should be close to {original_prob*100}%): {target_year['probability_percentage'].iloc[0]:.2f}%"
)

# %%
output = [39.715637, 43.708259, 46.858015, 49.230160, 50.954967]

# %%
