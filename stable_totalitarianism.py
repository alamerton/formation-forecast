# %%
from utils.logistic_interpolation import get_logistic_interpolation
from utils.geometric_mean_odds import get_geometric_mean
import pandas as pd

# %%
# Estimate Distribution for World Government

years = [2030, 2055, 2080, 2105, 2130]

k = 0.03  # Growth rate

original_year = 2100
original_prob = 0.06
gov_distribution = get_logistic_interpolation(years, original_year, original_prob, k)

# Print results
print("\nProbabilities for world government:")
print(gov_distribution.to_string(index=False))

# Verify we hit close to our target probability
target_year = get_logistic_interpolation(
    [original_year], original_year, original_prob, k
)
print(
    f"\nProbability for 2100 (should be close to {original_prob*100}%): {target_year['probability_percentage'].iloc[0]:.2f}%"
)

# %%
# Estimate Distribution for Stable Totalitarianism with Bryan Caplan's
# prection

years = [2030, 2055, 2080, 2105, 2130]

k = 0.0161  # Growth rate

original_year = 3011
original_prob = 0.05
bryan_distribution = get_logistic_interpolation(years, original_year, original_prob, k)

# Print results
print("\nProbabilities for stable totalitarianism:")
print(bryan_distribution.to_string(index=False))

# Verify we hit close to our target probability
target_year = get_logistic_interpolation(
    [original_year], original_year, original_prob, k
)
print(
    f"\nProbability for {original_year} (should be close to {original_prob*100}%): {target_year['probability_percentage'].iloc[0]:.2f}%"
)

# %%
# Estimate Distribution for Stable Totalitarianism with Stephen Clare's
# prection

years = [2030, 2055, 2080, 2105, 2130]

k = 0.03  # growth rate

original_year = 2124
original_prob = 0.0003
stephen_distribution = get_logistic_interpolation(
    years, original_year, original_prob, k
)

# Print results
print("\nProbabilities for stable totalitarianism:")
print(stephen_distribution.to_string(index=False))

# Verify we hit close to our target probability
target_year = get_logistic_interpolation(
    [original_year], original_year, original_prob, k
)
print(
    f"\nProbability for {original_year} (should be close to {original_prob*100}%): {target_year['probability_percentage'].iloc[0]:.2f}%"
)
# %%
# Get geometric mean of odds for Bryan and Stephen distributions
bryan_distribution = bryan_distribution.rename(
    columns={"probability_percentage": "bryan_probability_percentage"}
)
stephen_distribution = stephen_distribution.rename(
    columns={"probability_percentage": "stephen_probability_percentage"}
)

# Combine the DataFrames
combined_df = pd.merge(
    bryan_distribution[["year", "bryan_probability_percentage"]],
    stephen_distribution[["year", "stephen_probability_percentage"]],
    on="year",
)

combined_df["geometric_mean_odds"] = combined_df.apply(
    lambda row: get_geometric_mean(
        [row["bryan_probability_percentage"], row["stephen_probability_percentage"]]
    ),
    axis=1,
)

# Print the combined DataFrame
print("\nCombined Distribution with Geometric Mean of Odds:")
print(combined_df.to_string(index=False))

# %%
