# %% Imports
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
years = [2030, 2055, 2080, 2105, 2130]

stable_totalitarianism = [0.00000288, 0.00000406, 0.00000539, 0.00000689, 0.00000862]


# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(years), max(years), 500)

spline_stable_totalitarianism = make_interp_spline(years, stable_totalitarianism, k=3)
stable_totalitarianism_smooth = spline_stable_totalitarianism(years_smooth)


# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)
# Plotting the smooth curves
plt.plot(
    years_smooth,
    stable_totalitarianism_smooth,
    label="First Whole Brain Emulation",
    color="blue",
)


# Plotting the original data points

# Plotting the original data points
plt.scatter(
    years,
    stable_totalitarianism,
    label="Probabilities",
    color="purple",
    marker="x",
)

plt.title("Average Stable Totalitarianism Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 0.001)
plt.xticks(years)


plt.tight_layout()
plt.show()

# %%
