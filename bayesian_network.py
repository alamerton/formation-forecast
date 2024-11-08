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

    def calculate_conditional_probability(
        self,
        p_agi: List[float],
        p_misalignment: List[float],
        p_wwiii: List[float],
        p_wbe: List[float],
        p_stable_total: List[float],
        p_world_gov: List[float],
        cpt: Dict[Tuple[bool, bool, bool, bool, bool, bool], float],
    ) -> List[float]:
        p_lock_in = []

        for year_idx in range(len(self.forecast_years)):
            # Calculate joint probability across all possible combinations
            p_total = 0

            # Iterate through all possible combinations of events
            for agi in [True, False]:
                for misalignment in [True, False]:
                    for wwiii in [True, False]:
                        for wbe in [True, False]:
                            for stable_total in [True, False]:
                                for world_gov in [True, False]:
                                    # Calculate probability of this combination
                                    p_combination = (
                                        (
                                            p_agi[year_idx]
                                            if agi
                                            else (1 - p_agi[year_idx])
                                        )
                                        * (
                                            p_misalignment[year_idx]
                                            if misalignment
                                            else (1 - p_misalignment[year_idx])
                                        )
                                        * (
                                            p_wwiii[year_idx]
                                            if wwiii
                                            else (1 - p_wwiii[year_idx])
                                        )
                                        * (
                                            p_wbe[year_idx]
                                            if wbe
                                            else (1 - p_wbe[year_idx])
                                        )
                                        * (
                                            p_stable_total[year_idx]
                                            if stable_total
                                            else (1 - p_stable_total[year_idx])
                                        )
                                        * (
                                            p_world_gov[year_idx]
                                            if world_gov
                                            else (1 - p_world_gov[year_idx])
                                        )
                                    )

                                    # Multiply by conditional probability of lock-in given this combination
                                    p_total += (
                                        p_combination
                                        * cpt[
                                            (
                                                agi,
                                                misalignment,
                                                wwiii,
                                                wbe,
                                                stable_total,
                                                world_gov,
                                            )
                                        ]
                                    )

            p_lock_in.append(p_total)

        return p_lock_in

    def generate_forecast(
        self,
        p_agi: List[float],
        p_misalignment: float,
        p_wwiii: List[float],
        p_wbe: List[float],
        p_stable_total: List[float],
        p_world_gov: List[float],
        cpt: Dict[Tuple[bool, bool, bool, bool, bool, bool], float],
    ) -> Dict[str, List[float]]:

        # Calculate final lock-in probability
        p_lock_in = self.calculate_conditional_probability(
            p_agi, p_misalignment, p_wwiii, p_wbe, p_stable_total, p_world_gov, cpt
        )

        return {
            "years": self.forecast_years,
            "p_agi": p_agi,
            "p_wwiii": p_wwiii,
            "p_misalignment": [p_misalignment] * len(self.forecast_years),
            "p_wbe": p_wbe,
            "p_stable_total": p_stable_total,
            "p_world_gov": p_world_gov,
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
    p_agi = [0.234123, 0.684444, 0.809520, 0.962057, 0.991274]
    p_alignment = [
        0.39715637,
        0.43708259,
        0.46858015,
        0.49230160,
        0.50954967,
    ]
    p_wwiii = [0.3000, 0.3144, 0.3861, 0.4579, 0.5297]
    p_wbe = [0.0653, 0.402, 0.5218, 0.599, 0.6549]
    p_stable_total = [0.000288, 0.000405, 0.000539, 0.000689, 0.000861]
    p_world_gov = [0.0048, 0.0161, 0.04, 0.0639, 0.0752]

    # Conditional probability table
    # (AGI, Alignment, WWIII, WBE, Stable Totalitarianism, World Government) -> P(Lock-in)
    cpt = {
        (True, True, True, True, True, True): 0.51,
        (True, True, True, True, True, False): 0.50,
        (True, True, True, True, False, True): 0.41,
        (True, True, True, True, False, False): 0.40,
        (True, True, True, False, True, True): 0.41,
        (True, True, True, False, True, False): 0.40,
        (True, True, True, False, False, True): 0.31,
        (True, True, True, False, False, False): 0.30,
        (True, True, False, True, True, True): 0.41,
        (True, True, False, True, True, False): 0.40,
        (True, True, False, True, False, True): 0.31,
        (True, True, False, True, False, False): 0.30,
        (True, True, False, False, True, True): 0.31,
        (True, True, False, False, True, False): 0.30,
        (True, True, False, False, False, True): 0.21,
        (True, True, False, False, False, False): 0.20,
        (True, False, True, True, True, True): 0.41,
        (True, False, True, True, True, False): 0.40,
        (True, False, True, True, False, True): 0.31,
        (True, False, True, True, False, False): 0.30,
        (True, False, True, False, True, True): 0.31,
        (True, False, True, False, True, False): 0.30,
        (True, False, True, False, False, True): 0.21,
        (True, False, True, False, False, False): 0.20,
        (True, False, False, True, True, True): 0.31,
        (True, False, False, True, True, False): 0.30,
        (True, False, False, True, False, True): 0.21,
        (True, False, False, True, False, False): 0.20,
        (True, False, False, False, True, True): 0.21,
        (True, False, False, False, True, False): 0.20,
        (True, False, False, False, False, True): 0.11,
        (True, False, False, False, False, False): 0.10,
        (False, True, True, True, True, True): 0.41,
        (False, True, True, True, True, False): 0.40,
        (False, True, True, True, False, True): 0.31,
        (False, True, True, True, False, False): 0.30,
        (False, True, True, False, True, True): 0.31,
        (False, True, True, False, True, False): 0.30,
        (False, True, True, False, False, True): 0.21,
        (False, True, True, False, False, False): 0.20,
        (False, True, False, True, True, True): 0.31,
        (False, True, False, True, True, False): 0.30,
        (False, True, False, True, False, True): 0.21,
        (False, True, False, True, False, False): 0.20,
        (False, True, False, False, True, True): 0.21,
        (False, True, False, False, True, False): 0.20,
        (False, True, False, False, False, True): 0.11,
        (False, True, False, False, False, False): 0.10,
        (False, False, True, True, True, True): 0.31,
        (False, False, True, True, True, False): 0.30,
        (False, False, True, True, False, True): 0.21,
        (False, False, True, True, False, False): 0.20,
        (False, False, True, False, True, True): 0.21,
        (False, False, True, False, True, False): 0.20,
        (False, False, True, False, False, True): 0.11,
        (False, False, True, False, False, False): 0.10,
        (False, False, False, True, True, True): 0.21,
        (False, False, False, True, True, False): 0.20,
        (False, False, False, True, False, True): 0.11,
        (False, False, False, True, False, False): 0.10,
        (False, False, False, False, True, True): 0.11,
        (False, False, False, False, True, False): 0.10,
        (False, False, False, False, False, True): 0.01,
        (False, False, False, False, False, False): 0.00,
    }

    forecaster = LockInForecast()
    forecast = forecaster.generate_forecast(
        p_agi, p_alignment, p_wwiii, p_wbe, p_stable_total, p_world_gov, cpt
    )

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
