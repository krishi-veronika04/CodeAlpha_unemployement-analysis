import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import ttest_ind, mannwhitneyu
from datetime import datetime

# -----------------------------
# 1) Generate synthetic dataset
# -----------------------------
rng = pd.date_range(start="2010-01-01", end="2024-12-01", freq="MS")
n = len(rng)
np.random.seed(42)

trend = 8.5 - 0.01 * np.arange(n)
month_nums = rng.month
seasonal = 0.6 * np.sin(2 * np.pi * (month_nums / 12.0))

covid_shock = np.zeros(n)
for i, dt in enumerate(rng):
    if dt >= pd.Timestamp("2020-03-01") and dt <= pd.Timestamp("2021-12-01"):
        months_since = (dt.year - 2020) * 12 + dt.month - 3
        peak = 5.0 * np.exp(-0.25 * months_since) if months_since >= 0 else 0.0
        covid_shock[i] = peak

noise = np.random.normal(scale=0.35, size=n)
unemp = trend + seasonal + covid_shock + noise
unemp = np.round(unemp, 2)

df = pd.DataFrame({"date": rng, "unemployment_rate": unemp})
df.set_index("date", inplace=True)

# -----------------------------
# 2) Cleaning & basic checks
# -----------------------------
df.index = pd.to_datetime(df.index)
df["unemployment_rate"] = pd.to_numeric(df["unemployment_rate"], errors="coerce")
print("Missing values:", df["unemployment_rate"].isna().sum())
print(df["unemployment_rate"].describe())

# -----------------------------
# 3) Visualization
# -----------------------------
plt.figure(figsize=(11,4))
plt.plot(df.index, df["unemployment_rate"])
plt.title("Monthly Unemployment Rate")
plt.xlabel("Date")
plt.ylabel("Unemployment rate (%)")
plt.grid(True)
plt.show()

rolling_short = df["unemployment_rate"].rolling(window=3, center=True).mean()
rolling_long = df["unemployment_rate"].rolling(window=12, center=True).mean()

plt.figure(figsize=(11,4))
plt.plot(df.index, df["unemployment_rate"], label="Monthly")
plt.plot(rolling_short.index, rolling_short, label="3-month rolling")
plt.plot(rolling_long.index, rolling_long, label="12-month rolling")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 4) Seasonal decomposition
# -----------------------------
decomp = seasonal_decompose(df["unemployment_rate"], model="additive", period=12, extrapolate_trend='freq')
decomp.plot()
plt.show()

# -----------------------------
# 5) Year-over-year change
# -----------------------------
df_yoy = df["unemployment_rate"].pct_change(periods=12).dropna() * 100
print(df_yoy.resample("Y").mean())

monthly_avg = df["unemployment_rate"].groupby(df.index.month).mean()
print(monthly_avg)

# -----------------------------
# 6) Covid impact analysim  
# -----------------------------
pre = df.loc[:pd.Timestamp("2020-02-01"), "unemployment1()"]

 
