# %%
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from utils.geometric_mean_odds import get_geometric_mean


# %%
class LockInForecast:
    def __init__(self, forecast_years: List[int] = [2030, 2055, 2080, 2105, 2130]):
        self.forecast_years = forecast_years

    def calculate_agi_probability(
        self, agi_forecasts: List[List[float]]
    ) -> List[float]:
        """
        Combines multiple AGI timeline forecasts using geometric mean of
        odds
        Args:
            agi_forecasts: List of forecasts, each containing
            probabilities for forecast_years
        Returns:
            Combined AGI probability for each year
        """
        # Weights could be adjusted based on forecast quality/credibility
        # For each column (year) in agi forecasts, return the geometric
        # mean of odds of each value in that column (each forecast
        # representing that year)

        # [0.2809, 0.6433, 0.7649, 0.8673, 0.9250],  # Epoch model-based
        # [0.6572, 0.9334, 0.9603, 0.9697, 0.9745],  # Metaculus weakly general
        # [0.4183, 0.8478, 0.9002, 0.9279, 0.9552],  # Metaculus general
        # [0.1000, 0.5324, 0.8027, 1.0000, 1.0000],  # AI Impacts
        # [0.3100, 0.6480, 0.7380, 0.8280, 0.9180],  # Samotsvety

        weights = np.ones(len(agi_forecasts)) / len(agi_forecasts)
        # return np.average(agi_forecasts, weights=weights, axis=0)
        average_probs = []
        # For each distribution
        for col in zip(*agi_forecasts):
            average_probs[col] = get_geometric_mean(agi_forecasts[col])

        # Returns single distribution, average_probs, of shape (1,5) where thing[i] =
        # probability of agi for year i

    def calculate_conditional_probability(
        self,
        p_agi: List[float],
        p_misalignment: float,
        p_wwiii: List[float],
        cpt: Dict[Tuple[bool, bool, bool], float],
    ) -> List[float]:
        """
        Calculates lock-in probability using a simplified Bayesian network
        Args:
            p_agi: AGI probability for each year
            p_misalignment: Probability of AGI misalignment
            p_wwiii: WWIII probability for each year
            cpt: Conditional probability table for lock-in given AGI, misalignment, and WWIII
        Returns:
            Lock-in probability for each year
        """
        p_lock_in = []

        for year_idx in range(len(self.forecast_years)):
            # Calculate joint probability across all possible combinations
            p_total = 0

            # Iterate through all possible combinations of events
            for agi in [True, False]:
                for wwiii in [True, False]:
                    # Calculate probability of this combination
                    p_combination = (
                        (p_agi[year_idx] if agi else (1 - p_agi[year_idx]))
                        * (p_wwiii[year_idx] if wwiii else (1 - p_wwiii[year_idx]))
                        * (
                            p_misalignment if agi else 1.0
                        )  # Misalignment only matters if AGI exists (at least
                        # for the sake of this iteration)
                    )

                    # Multiply by conditional probability of lock-in given this combination
                    p_total += (
                        p_combination * cpt[(agi, True, wwiii)]
                    )  # Using True for misalignment when AGI exists

            p_lock_in.append(p_total)

        return p_lock_in

    def generate_forecast(
        self,
        agi_forecasts: List[List[float]],
        p_misalignment: float,
        p_wwiii: List[float],
        cpt: Dict[Tuple[bool, bool, bool], float],
    ) -> Dict[str, List[float]]:
        """
        Generates complete lock-in forecast
        Args:
            agi_forecasts: List of AGI timeline forecasts
            p_misalignment: Probability of AGI misalignment
            p_wwiii: WWIII probability for each year
            cpt: Conditional probability table
        Returns:
            Dictionary containing both input probabilities and final forecast
        """
        # Combine AGI forecasts
        p_agi = self.calculate_agi_probability(agi_forecasts)

        # Calculate final lock-in probability
        p_lock_in = self.calculate_conditional_probability(
            p_agi, p_misalignment, p_wwiii, cpt
        )

        return {
            "years": self.forecast_years,
            "p_agi": p_agi.tolist(),
            "p_wwiii": p_wwiii,
            "p_misalignment": p_misalignment,
            "p_lock_in": p_lock_in,
        }


# %%


def plot_forecast(forecast):
    years = forecast["years"]
    p_agi = forecast["p_agi"]
    p_wwiii = forecast["p_wwiii"]
    p_lock_in = forecast["p_lock_in"]

    # Create smooth curves using spline interpolation
    years_smooth = np.linspace(min(years), max(years), 500)  # 500 points for smoothness

    # Interpolating AGI Probability
    spline_agi = make_interp_spline(years, p_agi, k=3)  # Cubic spline
    p_agi_smooth = spline_agi(years_smooth)

    # Interpolating WWIII Probability
    spline_wwiii = make_interp_spline(years, p_wwiii, k=3)  # Cubic spline
    p_wwiii_smooth = spline_wwiii(years_smooth)

    # Interpolating Lock-in Probability
    spline_lock_in = make_interp_spline(years, p_lock_in, k=3)  # Cubic spline
    p_lock_in_smooth = spline_lock_in(years_smooth)

    plt.figure(figsize=(12, 6))

    # Plotting the smooth curves
    plt.plot(years_smooth, p_agi_smooth, label="AGI Probability", color="blue")
    plt.plot(years_smooth, p_wwiii_smooth, label="WWIII Probability", color="orange")
    plt.plot(years_smooth, p_lock_in_smooth, label="Lock-in Probability", color="green")

    # Plotting the original data points
    plt.scatter(years, p_agi, color="blue", marker="o")  # AGI points
    plt.scatter(years, p_wwiii, color="orange", marker="s")  # WWIII points
    plt.scatter(years, p_lock_in, color="green", marker="^")  # Lock-in points

    plt.title("Lock-in Forecast Over Time")
    plt.xlabel("Year")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.ylim(0, 1)  # Set y-axis limits from 0 to 1
    plt.xticks(years)  # Set x-axis ticks to the forecast years

    # # Add value labels at the original data points
    # for i, year in enumerate(years):
    #     plt.text(year, p_agi[i], f"{p_agi[i]:.2f}", ha="center", va="bottom")
    #     plt.text(year, p_wwiii[i], f"{p_wwiii[i]:.2f}", ha="center", va="bottom")
    #     plt.text(year, p_lock_in[i], f"{p_lock_in[i]:.2f}", ha="center", va="top")

    plt.tight_layout()
    plt.show()


# %%
def main():
    agi_forecasts = [
        [0.2809, 0.6433, 0.7649, 0.8673, 0.9250],  # Epoch model-based
        [0.6572, 0.9334, 0.9603, 0.9697, 0.9745],  # Metaculus weakly general
        [0.4183, 0.8478, 0.9002, 0.9279, 0.9552],  # Metaculus general
        [0.1000, 0.5324, 0.8027, 1.0000, 1.0000],  # AI Impacts
        [0.3100, 0.6480, 0.7380, 0.8280, 0.9180],  # Samotsvety
    ]
    alignment_difficulty_vals = [
        0.4,
        0.15,
        0.75,
        0.65,
        0.75,
        0.7,
        0.95,
        0.5,
        0.001,
        0.3,
        0.8,
    ]

    p_misalignment = get_geometric_mean(alignment_difficulty_vals)

    p_wwiii = [0.3000, 0.3144, 0.3861, 0.4579, 0.5297]  # Interpolated/extrapolated

    # Sample conditional probability table
    # (AGI, Misaligned, WWIII) -> P(Lock-in)
    cpt = {
        (True, True, True): 0.95,  # Very high chance with both AGI and WWIII
        (True, True, False): 0.80,  # High chance with misaligned AGI alone
        (True, False, True): 0.70,  # High chance with aligned AGI and WWIII
        (True, False, False): 0.30,  # Lower chance with aligned AGI alone
        (False, True, True): 0.60,  # Moderate chance with WWIII alone
        (False, True, False): 0.15,  # Lower without any factors
        (False, False, True): 0.60,  # As above
        (False, False, False): 0.15,  # As above
    }

    forecaster = LockInForecast()
    forecast = forecaster.generate_forecast(agi_forecasts, p_misalignment, p_wwiii, cpt)

    print("\nLock-in Forecast Results:")
    print("-----------------------")
    for i, year in enumerate(forecast["years"]):
        print(f"Year {year}:")
        print(f"  AGI Probability: {forecast['p_agi'][i]:.2%}")
        print(f"  WWIII Probability: {forecast['p_wwiii'][i]:.2%}")
        print(f"  Lock-in Probability: {forecast['p_lock_in'][i]:.2%}")
        print()

    # Plot lock-in probability distribution
    plot_forecast(forecast)


# %%
if __name__ == "__main__":
    main()

# %%
