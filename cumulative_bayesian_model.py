# %%
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

time_points = np.array([2030, 2055, 2080, 2105, 2130])

# %%
agi_cdf = np.array([0.2809, 0.6433, 0.7649, 0.8673, 0.9250])
misaligned_agi = 0.5293
wwiii_cdf = np.array([0.3, 0.3144, 0.3861, 0.4579, 0.5297])


# %%
## AGI Probability Distribution
years = np.linspace(2020, 2140, 1000)

plt.figure(figsize=(12, 8))
plt.plot(years, agi_cdf, label="AGI", color="blue")
plt.plot(years, wwiii_cdf, label="WWIII", color="red")

plt.scatter(time_points, agi_cdf, color="blue")
plt.scatter(time_points, wwiii_cdf, color="red")

plt.title("AGI and WWIII Cumulative Probability Distributions Over Time", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Cumulative Probability", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)

plt.ylim(0, 1)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()


# %%
## Fit a normal distribution to the probabilities
def fit_normal_distribution(cdf, time_points):
    def objective(params):
        mu, sigma = params
        return np.sum((norm.cdf(time_points, mu, sigma) - cdf) ** 2)

    from scipy.optimize import minimize

    result = minimize(objective, [2080, 50])
    return result.x


agi_params = fit_normal_distribution(agi_cdf, time_points)
wwiii_params = fit_normal_distribution(wwiii_cdf, time_points)


# %%
## Calculate probability using cumulative probabilities
def calculate_lock_in_prob(agi_prob, wwiii_prob):
    # TODO: Placeholder, replace with weightings
    # also add misalignment probability to the calculation

    return max(agi_prob, wwiii_prob)


# %%
def lock_in_probability(year):
    agi_prob = norm.cdf(year, *agi_params)
    wwiii_prob = norm.cdf(year, *wwiii_params)
    return calculate_lock_in_prob(agi_prob, wwiii_prob)


years = np.linspace(2020, 2140, 1000)

agi_probs = norm.cdf(years, *agi_params)
wwiii_probs = norm.cdf(years, *wwiii_params)
lock_in_probs = [lock_in_probability(year) for year in years]
# %%

# lock_in_probabilities = [lock_in_probability(i) for i in time_points]

for year in time_points:
    prob = lock_in_probability(year)
    print(f"Probability of lock-in by {year}: {prob:.4f}")

# %%
## Display distribution as graph
plt.figure(figsize=(12, 8))
plt.plot(years, agi_probs, label="AGI", color="blue")
plt.plot(years, wwiii_probs, label="WWIII", color="red")
plt.plot(years, lock_in_probs, label="Lock-in", color="purple", linestyle="--")

plt.scatter(time_points, agi_cdf, color="blue", zorder=5)
plt.scatter(time_points, wwiii_cdf, color="red", zorder=5)
lock_in_points = [lock_in_probability(year) for year in time_points]
plt.scatter(time_points, lock_in_points, color="purple", zorder=5)

plt.title("Cumulative Probability Distributions")
plt.xlabel("Year")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid(True)
plt.ylim(0, 1)
plt.show()

# %%
