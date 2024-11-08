# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
original_years = [2030, 2050, 2100]
new_years = [2030, 2055, 2080, 2105, 2130]

original_probabilities = [0.31, 0.63, 0.81]
probabilities_int_ext = [0.31, 0.648, 0.738, 0.828, 0.918]

# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(new_years), max(new_years), 500)

spline_probabilities_int_ext = make_interp_spline(new_years, probabilities_int_ext, k=3)
probabilities_int_ext_smooth = spline_probabilities_int_ext(years_smooth)

# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)

# Plotting the smooth curves
plt.plot(
    years_smooth,
    probabilities_int_ext_smooth,
    color="blue",
)

plt.scatter(
    original_years,
    original_probabilities,
    label="Original Probabilities",
    color="orange",
    marker="x",
)

plt.scatter(
    new_years,
    probabilities_int_ext,
    label="Interpolated & Extrapolated Probabilities",
    color="purple",
    marker="x",
)

plt.title("Samotsvety AGI Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.xticks(new_years)

plt.tight_layout()
plt.show()
