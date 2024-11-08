# %%
import pandas as pd
import numpy as np


def get_logistic_interpolation(years, target_year, target_prob, k):
    """
    Calculate probability of an event across different time horizons
    using logistic interpolation.

    Args:
        years (list): List of years to calculate probabilities for
        target_prob (float): Known probability for target year (default 0.06 for 2100)
        target_year (int): Year for which we know the probability (default 2100)

    Returns:
        pandas.DataFrame: Years and their corresponding probabilities
    """
    # Logistic function parameters
    L = target_prob * 1.2  # maximum probability cap (slightly above target_prob)
    x0 = target_year - 100  # midpoint year
    base_year = 2000  # reference year for calculation

    # Calculate probabilities using logistic function
    probabilities = []
    for year in years:
        t = year - base_year
        t0 = x0 - base_year
        p = L / (1 + np.exp(-k * (t - t0)))
        probabilities.append(p)

    # Create DataFrame with results
    results = pd.DataFrame(
        {
            "year": years,
            "probability": probabilities,
            "probability_percentage": [p * 100 for p in probabilities],
        }
    )

    return results
