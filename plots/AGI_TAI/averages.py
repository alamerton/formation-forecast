# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# %% Set probability values
years = [2030, 2055, 2080, 2105, 2130]

agi = [0.234123, 0.684444, 0.809520, 0.962057, 0.991274]


# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(years), max(years), 500)

# spline_agi = make_interp_spline(years, agi, k=3)
# agi_smooth = spline_agi(years_smooth)

pchip_agi = PchipInterpolator(years, agi)
agi_smooth = np.clip(pchip_agi(years_smooth), 0, 1)  # Clip values to [0, 1]


# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)
# Plotting the smooth curves
plt.plot(
    years_smooth,
    agi_smooth,
    label="First Whole Brain Emulation",
    color="blue",
)


# Plotting the original data points

# Plotting the original data points
plt.scatter(
    years,
    agi,
    label="Probabilities",
    color="purple",
    marker="x",
)

plt.title("Average AGI Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.xticks(years)


plt.tight_layout()
plt.show()

# %%
