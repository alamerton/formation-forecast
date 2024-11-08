# %% Imports
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
original_years = [2030, 2050, 2100]
new_years = [2030, 2055, 2080, 2105, 2130]

model_based_averages = [0.08, 0.27, 0.54]
model_based_averages_int_ext = [0.08, 0.297, 0.432, 0.567, 0.702]

judgement_based_averages = [0.12, 0.57, 0.88]
judgement_based_averages_int_ext = [0.12, 0.601, 0.756, 0.911, 1]

# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(new_years), max(new_years), 500)

spline_model_based = make_interp_spline(new_years, model_based_averages_int_ext, k=3)
model_based_int_ext_smooth = spline_model_based(years_smooth)

spline_judgement_based = make_interp_spline(
    new_years, judgement_based_averages_int_ext, k=3
)
judgement_based_int_ext_smooth = spline_judgement_based(years_smooth)

# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)


# Plotting the smooth curves
plt.plot(
    years_smooth,
    model_based_int_ext_smooth,
    label="Model-Based Averages",
    color="blue",
)

plt.plot(
    years_smooth,
    judgement_based_int_ext_smooth,
    label="Judgement-Based Averages",
    color="green",
)


# Plotting the original data points
plt.scatter(
    original_years,
    model_based_averages,
    label="Original Averages",
    color="orange",
    marker="x",
)
plt.scatter(
    new_years,
    model_based_averages_int_ext,
    label="Interpolated & Extrapolated Averages",
    color="purple",
    marker="x",
)
plt.scatter(
    original_years,
    judgement_based_averages,
    color="orange",
    marker="x",
)
plt.scatter(
    new_years,
    judgement_based_averages_int_ext,
    color="purple",
    marker="x",
)

plt.title("Epoch Literature Review AGI/TAI Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.xticks(new_years)

plt.tight_layout()
plt.show()

# %%
