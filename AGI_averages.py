# %%
import pandas as pd
from utils.geometric_mean_odds import get_geometric_mean, odds_to_probability

arrays = [
    [2030, 2055, 2080, 2105, 2130],  # Year
    [0.08, 0.297, 0.432, 0.567, 0.702],  # Epoch model-based
    [0.12, 0.601, 0.756, 0.911, 0.9999],  # Epoch judgement-based
    [0.6572, 0.9334, 0.9603, 0.9697, 0.9745],  # Metaculus weakly general
    [0.4183, 0.8478, 0.9002, 0.9279, 0.9552],  # Metaculus general
    [0.1, 0.5324, 0.8027, 0.9999, 0.9999],  # AI Impacts HLMI
    [0.31, 0.648, 0.738, 0.828, 0.918],  # Samotsvety
]

# Define column names (these will be the index in the final DataFrame)
index = [
    "Year",
    "Epoch model-based",
    "Epoch judgement-based",
    "Metaculus weakly general",
    "Metaculus general",
    "AI Impacts HLMI",
    "Samotsvety",
]

# Create DataFrame
df = pd.DataFrame(arrays, index=index)
df = df.T

print(df.to_string(index=False))

# %%
df["geometric_mean_odds"] = df.apply(
    lambda row: odds_to_probability(
        get_geometric_mean(
            [
                row["Epoch model-based"],
                row["Epoch judgement-based"],
                row["Metaculus weakly general"],
                row["Metaculus general"],
                row["AI Impacts HLMI"],
                row["Samotsvety"],
            ]
        )
    ),
    axis=1,
)
print(df.to_string(index=False))  # %%
# 0.234123, 0.684444, 0.809520, 0.845268, 0.840579

# %%
print(df["geometric_mean_odds"].to_string(index=False))

# [0.234123, 0.684444, 0.809520, 0.962057, 0.991274]

# %%
