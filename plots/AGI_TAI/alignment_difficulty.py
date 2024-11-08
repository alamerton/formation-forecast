# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# Data
probabilities = [0.4, 0.15, 0.75, 0.65, 0.75, 0.7, 0.95, 0.5, 0.001, 0.3, 0.8, 0.4569]
names = [
    "Leopold Aschenbrenner",
    "Ben Garfinkel",
    "Daniel Kokotajlo",
    "Ben Levinstein",
    "Eli Lifland",
    "Neel Nanda",
    "Nate Soares",
    "Christian Tarsney",
    "David Thorstad",
    "David Wallace",
    "Anonymous 1",
    "Average",
]

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(names, probabilities, color="#6D5ACF")

ax.set_ylabel("Probability", fontsize=12)
ax.set_title("Alignment Difficulty Predictions", fontsize=14)
ax.set_ylim(0, 1)

plt.xticks(rotation=45, ha="right")

for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{height:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
plt.show()
# %% Imports

# %% Set probability values
years = [2030, 2055, 2080, 2105, 2130]

alignment_difficulty = [0.39715637, 0.43708259, 0.46858015, 0.49230160, 0.50954967]


# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(years), max(years), 500)

spline_alignment_difficulty = make_interp_spline(years, alignment_difficulty, k=3)
alignment_difficulty_smooth = spline_alignment_difficulty(years_smooth)


# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)
# Plotting the smooth curves
plt.plot(
    years_smooth,
    alignment_difficulty_smooth,
    label="Alignment Difficulty",
    color="blue",
)

# Plotting the original data points
plt.scatter(
    years,
    alignment_difficulty,
    label="Probabilities",
    color="purple",
    marker="x",
)

plt.title("Interpolated Alignment Difficulty Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.xticks(years)


plt.tight_layout()
plt.show()

# %%
