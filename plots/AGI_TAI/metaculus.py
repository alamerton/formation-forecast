# %% Imports
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# %% Set probability values
years = [2030, 2055, 2080, 2105, 2130]

weakly_general = [0.6572, 0.9334, 0.9603, 0.9697, 0.9745]
general = [0.4183, 0.8478, 0.9002, 0.9279, 0.9552]

# %% Create smooth curves using spline interpolation
years_smooth = np.linspace(min(years), max(years), 500)

spline_weakly_general = make_interp_spline(years, weakly_general, k=3)
weakly_general_smooth = spline_weakly_general(years_smooth)

spline_general = make_interp_spline(years, general, k=3)
general_smooth = spline_general(years_smooth)

# %% Create plot
plt.figure(figsize=(6, 4), dpi=300)
# Plotting the smooth curves
plt.plot(
    years_smooth,
    weakly_general_smooth,
    label="Weakly General AI",
    color="blue",
)

plt.plot(
    years_smooth,
    general_smooth,
    label="General AI",
    color="green",
)


# Plotting the original data points

# Plotting the original data points
plt.scatter(
    years,
    weakly_general,
    label="Probabilities",
    color="purple",
    marker="x",
)
# plt.scatter(
#     years,
#     model_based_averages_int_ext,
#     label="Interpolated & Extrapolated Averages",
#     color="purple",
#     marker="x",
# )
plt.scatter(
    years,
    general,
    color="purple",
    marker="x",
)
# plt.scatter(
#     years,
#     judgement_based_averages_int_ext,
#     color="purple",
#     marker="x",
# )

plt.title("Metaculus AGI Probabilities")
plt.xlabel("Year")
plt.ylabel("Probability")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.xticks(years)


plt.tight_layout()
plt.show()

# %%
