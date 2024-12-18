# %% Imports
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
years = [2030, 2055, 2080, 2105, 2130]

wbe = [0.0653, 0.402, 0.5218, 0.599, 0.6549]

# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(years), max(years), 500)

spline_wbe = make_interp_spline(years, wbe, k=3)
wbe_smooth = spline_wbe(years_smooth)


# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)
# Plotting the smooth curves
plt.plot(
    years_smooth,
    wbe_smooth,
    label="First Whole Brain Emulation",
    color="blue",
)


# Plotting the original data points

# Plotting the original data points
plt.scatter(
    years,
    wbe,
    label="Probabilities",
    color="purple",
    marker="x",
)

plt.title("Metaculus Whole Brain Emulation Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.xticks(years)


plt.tight_layout()
plt.show()

# %%
