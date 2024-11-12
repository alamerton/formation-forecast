# %% Imports
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
years = [2030, 2055, 2080, 2105, 2130]

stable_totalitarianism = [0.000288, 0.000405, 0.000539, 0.000689, 0.000861]


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
    label="Stable Totalitarianism",
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

plt.ylim(0, 0.01)
plt.xticks(years)


plt.tight_layout()
plt.show()

# %%
