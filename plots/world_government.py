# %% Imports
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
years = [2030, 2055, 2080, 2105, 2130]

world_government = [0.0048, 0.0161, 0.04, 0.0639, 0.0752]


# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(years), max(years), 500)

spline_world_government = make_interp_spline(years, world_government, k=3)
world_government_smooth = spline_world_government(years_smooth)


# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)
# Plotting the smooth curves
plt.plot(
    years_smooth,
    world_government_smooth,
    label="First Whole Brain Emulation",
    color="blue",
)


# Plotting the original data points

# Plotting the original data points
plt.scatter(
    years,
    world_government,
    label="Probabilities",
    color="purple",
    marker="x",
)

plt.title("Interpolated World Government Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.xticks(years)


plt.tight_layout()
plt.show()

# %%
