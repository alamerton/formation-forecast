# %%
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


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
        p_misalignment: List[float],
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
            "p_misalignment": p_misalignment,
            "p_wbe": p_wbe,
            "p_stable_total": p_stable_total,
            "p_world_gov": p_world_gov,
            "p_lock_in": p_lock_in,
        }


# %%


def plot_forecast(forecast):
    years = forecast["years"]

    probabilities = {
        "AGI": forecast["p_agi"],
        "WWIII": forecast["p_wwiii"],
        "Misalignment": forecast["p_misalignment"],
        "WBE": forecast["p_wbe"],
        "Stable Total.": forecast["p_stable_total"],
        "World Gov.": forecast["p_world_gov"],
        "Lock-in": forecast["p_lock_in"],
    }

    years_smooth = np.linspace(min(years), max(years), 500)

    plt.figure(figsize=(6, 4))

    for label, values in probabilities.items():
        if label != "Lock-in":
            spline = PchipInterpolator(years, values)
            values_smooth = spline(years_smooth)

            plt.plot(
                years_smooth, values_smooth, color="grey", alpha=0.2, linestyle="--"
            )
            plt.scatter(years, values, color="grey", alpha=0.2, marker="o", s=20)

    spline_lock_in = PchipInterpolator(years, probabilities["Lock-in"])
    p_lock_in_smooth = spline_lock_in(years_smooth)

    plt.plot(
        years_smooth,
        p_lock_in_smooth,
        label="Lock-in",
        color="green",
        linewidth=2,
    )
    plt.scatter(
        years,
        probabilities["Lock-in"],
        label="Probabilities",
        color="green",
        marker="^",
        s=50,
    )

    plt.title("Lock-in Forecast Plotted with Conditional Probabilities")
    plt.xlabel("Year")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.ylim(0, 1)
    plt.xticks(years)

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

    # Conditional Probability Weights
    agi_weight = 0.1
    alignment_weight = 0.1
    wwiii_weight = 0.1
    wbe_weight = 0.1
    stable_total_weight = 0.1
    world_gov_weight = 0.1

    # Conditional probability table
    # (AGI, Alignment, WWIII, WBE, Stable Totalitarianism, World Government) -> P(Lock-in)
    cpt = {
        (True, True, True, True, True, True): sum(
            [
                agi_weight,
                alignment_weight,
                wwiii_weight,
                wbe_weight,
                stable_total_weight,
                world_gov_weight,
            ]
        ),
        (True, True, True, True, True, False): sum(
            [
                agi_weight,
                alignment_weight,
                wwiii_weight,
                wbe_weight,
                stable_total_weight,
            ]
        ),
        (True, True, True, True, False, True): sum(
            [agi_weight, alignment_weight, wwiii_weight, wbe_weight, world_gov_weight]
        ),
        (True, True, True, True, False, False): sum(
            [agi_weight, alignment_weight, wwiii_weight, wbe_weight]
        ),
        (True, True, True, False, True, True): sum(
            [
                agi_weight,
                alignment_weight,
                wwiii_weight,
                stable_total_weight,
                world_gov_weight,
            ]
        ),
        (True, True, True, False, True, False): sum(
            [agi_weight, alignment_weight, wwiii_weight, stable_total_weight]
        ),
        (True, True, True, False, False, True): sum(
            [agi_weight, alignment_weight, wwiii_weight, world_gov_weight]
        ),
        (True, True, True, False, False, False): sum(
            [agi_weight, alignment_weight, wwiii_weight]
        ),
        (True, True, False, True, True, True): sum(
            [
                agi_weight,
                alignment_weight,
                wbe_weight,
                stable_total_weight,
                world_gov_weight,
            ]
        ),
        (True, True, False, True, True, False): sum(
            [agi_weight, alignment_weight, wbe_weight, stable_total_weight]
        ),
        (True, True, False, True, False, True): sum(
            [agi_weight, alignment_weight, wbe_weight, world_gov_weight]
        ),
        (True, True, False, True, False, False): sum(
            [agi_weight, alignment_weight, wbe_weight]
        ),
        (True, True, False, False, True, True): sum(
            [agi_weight, alignment_weight, stable_total_weight, world_gov_weight]
        ),
        (True, True, False, False, True, False): sum(
            [agi_weight, alignment_weight, stable_total_weight]
        ),
        (True, True, False, False, False, True): sum(
            [agi_weight, alignment_weight, world_gov_weight]
        ),
        (True, True, False, False, False, False): sum([agi_weight, alignment_weight]),
        (True, False, True, True, True, True): sum(
            [
                agi_weight,
                wwiii_weight,
                wbe_weight,
                stable_total_weight,
                world_gov_weight,
            ]
        ),
        (True, False, True, True, True, False): sum(
            [agi_weight, wwiii_weight, wbe_weight, stable_total_weight]
        ),
        (True, False, True, True, False, True): sum(
            [agi_weight, wwiii_weight, wbe_weight, world_gov_weight]
        ),
        (True, False, True, True, False, False): sum(
            [agi_weight, wwiii_weight, wbe_weight]
        ),
        (True, False, True, False, True, True): sum(
            [agi_weight, wwiii_weight, stable_total_weight, world_gov_weight]
        ),
        (True, False, True, False, True, False): sum(
            [agi_weight, wwiii_weight, stable_total_weight]
        ),
        (True, False, True, False, False, True): sum(
            [agi_weight, wwiii_weight, world_gov_weight]
        ),
        (True, False, True, False, False, False): sum([agi_weight, wwiii_weight]),
        (True, False, False, True, True, True): sum(
            [agi_weight, wbe_weight, stable_total_weight, world_gov_weight]
        ),
        (True, False, False, True, True, False): sum(
            [agi_weight, wbe_weight, stable_total_weight]
        ),
        (True, False, False, True, False, True): sum(
            [agi_weight, wbe_weight, world_gov_weight]
        ),
        (True, False, False, True, False, False): sum([agi_weight, wbe_weight]),
        (True, False, False, False, True, True): sum(
            [agi_weight, stable_total_weight, world_gov_weight]
        ),
        (True, False, False, False, True, False): sum(
            [agi_weight, stable_total_weight]
        ),
        (True, False, False, False, False, True): sum([agi_weight, world_gov_weight]),
        (True, False, False, False, False, False): agi_weight,
        (False, True, True, True, True, True): sum(
            [
                alignment_weight,
                wwiii_weight,
                wbe_weight,
                stable_total_weight,
                world_gov_weight,
            ]
        ),
        (False, True, True, True, True, False): sum(
            [alignment_weight, wwiii_weight, wbe_weight, stable_total_weight]
        ),
        (False, True, True, True, False, True): sum(
            [alignment_weight, wwiii_weight, wbe_weight, world_gov_weight]
        ),
        (False, True, True, True, False, False): sum(
            [alignment_weight, wwiii_weight, wbe_weight]
        ),
        (False, True, True, False, True, True): sum(
            [alignment_weight, wwiii_weight, stable_total_weight, world_gov_weight]
        ),
        (False, True, True, False, True, False): sum(
            [alignment_weight, wwiii_weight, stable_total_weight]
        ),
        (False, True, True, False, False, True): sum(
            [alignment_weight, wwiii_weight, world_gov_weight]
        ),
        (False, True, True, False, False, False): sum([alignment_weight, wwiii_weight]),
        (False, True, False, True, True, True): sum(
            [alignment_weight, wbe_weight, stable_total_weight, world_gov_weight]
        ),
        (False, True, False, True, True, False): sum(
            [alignment_weight, wbe_weight, stable_total_weight]
        ),
        (False, True, False, True, False, True): sum(
            [alignment_weight, wbe_weight, world_gov_weight]
        ),
        (False, True, False, True, False, False): sum([alignment_weight, wbe_weight]),
        (False, True, False, False, True, True): sum(
            [alignment_weight, stable_total_weight, world_gov_weight]
        ),
        (False, True, False, False, True, False): sum(
            [alignment_weight, stable_total_weight]
        ),
        (False, True, False, False, False, True): sum(
            [alignment_weight, world_gov_weight]
        ),
        (False, True, False, False, False, False): alignment_weight,
        (False, False, True, True, True, True): sum(
            [wwiii_weight, wbe_weight, stable_total_weight, world_gov_weight]
        ),
        (False, False, True, True, True, False): sum(
            [wwiii_weight, wbe_weight, stable_total_weight]
        ),
        (False, False, True, True, False, True): sum(
            [wwiii_weight, wbe_weight, world_gov_weight]
        ),
        (False, False, True, True, False, False): sum([wwiii_weight, wbe_weight]),
        (False, False, True, False, True, True): sum(
            [wwiii_weight, stable_total_weight, world_gov_weight]
        ),
        (False, False, True, False, True, False): sum(
            [wwiii_weight, stable_total_weight]
        ),
        (False, False, True, False, False, True): sum([wwiii_weight, world_gov_weight]),
        (False, False, True, False, False, False): wwiii_weight,
        (False, False, False, True, True, True): sum(
            [wbe_weight, stable_total_weight, world_gov_weight]
        ),
        (False, False, False, True, True, False): sum(
            [wbe_weight, stable_total_weight]
        ),
        (False, False, False, True, False, True): sum([wbe_weight, world_gov_weight]),
        (False, False, False, True, False, False): wbe_weight,
        (False, False, False, False, True, True): sum(
            [stable_total_weight, world_gov_weight]
        ),
        (False, False, False, False, True, False): stable_total_weight,
        (False, False, False, False, False, True): world_gov_weight,
        (False, False, False, False, False, False): 0.0,
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
