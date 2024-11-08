# %%
import math

# %%


def get_odds(p):
    if p == 1:
        return p
    return p / (1 - p)


def get_geometric_mean(values):
    odds_values = [get_odds(v) for v in values]
    n = len(odds_values)
    return (math.prod(odds_values)) ** (1 / n)
    # return np.exp(np.mean(np.log(values)))


def odds_to_probability(odds):
    return odds / (1 + odds)
