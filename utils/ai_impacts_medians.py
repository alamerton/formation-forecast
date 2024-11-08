# %%
import pandas as pd

# %%
## Get AI impacts distributions
df = pd.read_csv("../data/ai_impacts.csv")
print(df.head)

# %%
## Convert to numeric values

df = df.apply(pd.to_numeric, errors="coerce")
print(df.head)

# %%
## Handle special values
df = df.replace("never", float("inf"))
print(df.head)

# %%
## Remove any row that has no values for any of the columns
df = df.dropna(how="all")
print(df.head)

# %%
median_10_percent = df["years_until_10_percent"].median()
median_50_percent = df["years_until_50_percent"].median()
median_90_percent = df["years_until_90_percent"].median()

print(f"Median number of years for 10% probability of HLMI: {median_10_percent}")
print(f"Median number of years for 50% probability of HLMI: {median_50_percent}")
print(f"Median number of years for 90% probability of HLMI: {median_90_percent}")

# %%
## Calculate year after AI Impacts survey each median represents
print(f"AI Impacts simplistic probability of HLMI by {2022 + median_10_percent}: 10%")
print(f"AI Impacts simplistic probability of HLMI by {2022 + median_50_percent}: 50%")
print(f"AI Impacts simplistic probability of HLMI by {2022 + median_90_percent}: 90%")
# %%
