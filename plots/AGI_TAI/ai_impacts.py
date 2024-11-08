# %% Imports
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
original_years = [2059]
median_years = [2032, 2052, 2089]
new_years = [2030, 2055, 2080, 2105]
medians = [0.1, 0.5, 0.9]

# original_probability = [0.5]
probabilities_int_ext = [0.1, 0.5324, 0.8027, 1]

# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(new_years), new_years[-1], 500)

spline_probabilities = make_interp_spline(new_years, probabilities_int_ext, k=3)
probabilities_int_ext_smooth = spline_probabilities(years_smooth)

# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)

# Plotting the smooth curves
plt.plot(
    years_smooth,
    probabilities_int_ext_smooth,
    color="blue",
)

# Plotting the original data points
plt.scatter(
    median_years,
    medians,
    label="Median Survey Probabilities",
    color="green",
    marker="x",
)

plt.scatter(
    new_years,
    probabilities_int_ext,
    label="Interpolated & Extrapolated Probabilities",
    color="purple",
    marker="x",
)

plt.title("AI Impacts HLMI Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1.1)
plt.xticks(new_years)

plt.tight_layout()
plt.show()

# %%
