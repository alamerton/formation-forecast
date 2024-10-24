# %%
import numpy as np


# %%
def interpolate(x, x_values, y_values):
    return np.interp(x, x_values, y_values)


# %%
def extrapolate(x, x_values, y_values):
    slope = (y_values[-1] - y_values[-2]) / (x_values[-1] - x_values[-2])
    return y_values[-1] + slope * (x - x_values[-1])


# %%
x_values = [2050, 2151]
y_values = [30, 59]  # Percentages

# %%
year_a = 2030
year_b = 2055
year_c = 2080
year_d = 2105
year_e = 2130

# Interpolation
print(f"Interpolated value for {year_a}: {interpolate(year_a, x_values, y_values)}%")
print(f"Interpolated value for {year_b}: {interpolate(year_b, x_values, y_values)}%")
print(f"Interpolated value for {year_c}: {interpolate(year_c, x_values, y_values)}%")

# Extrapolation
print(f"Extrapolated value at {year_d}: {extrapolate(year_d, x_values, y_values)}%")
print(f"Extrapolated value at {year_e}: {extrapolate(year_e, x_values, y_values)}%")

# %%
