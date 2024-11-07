# %%
import math

values = [0.4, 0.15, 0.75, 0.65, 0.75, 0.7, 0.95, 0.5, 0.001, 0.3, 0.8]


def odds(p):
    return p / (1 - p)


def geometric_mean(values):
    odds_values = [odds(v) for v in values]
    print(odds_values)
    n = len(odds_values)
    return (math.prod(odds_values)) ** (1 / n)


# %%
geometric_mean_value = geometric_mean(values)
print(f"The geometric mean of the odds is: {geometric_mean_value:.4f}")
