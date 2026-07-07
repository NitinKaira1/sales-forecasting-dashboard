"""
Terminal-only version — graphs open ONE BY ONE.
Close each graph window to move to the next step (plt.show() blocks
execution until you close the window — that's what gives us the
"one at a time" behavior).

Run with:
    python terminal_demo.py
"""

import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# STEP 1: Generate sample sales data
# ---------------------------------------------------------------------------
def generate_sample_sales_data(start_date="2022-01-01", periods=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")
    t = np.arange(periods)

    trend = 200 + 0.15 * t
    yearly_seasonality = 40 * np.sin(2 * np.pi * (t - 80) / 365)
    day_of_week = dates.dayofweek
    weekly_seasonality = np.where(day_of_week >= 5, 25, 0)
    noise = rng.normal(loc=0, scale=15, size=periods)

    y = trend + yearly_seasonality + weekly_seasonality + noise
    y = np.maximum(y, 0)

    return pd.DataFrame({"ds": dates, "y": y})


print("STEP 1: Generating sample sales data...")
df = generate_sample_sales_data()
print(df.head())
print(f"... total rows: {len(df)}")
print("-" * 60)

# ---- GRAPH 1: Just the raw data, before any model touches it ----
print("Showing GRAPH 1: raw input data. Close the window to continue...")
plt.figure(figsize=(12, 5))
plt.plot(df["ds"], df["y"], color="black", linewidth=1)
plt.title("Raw Sales Data (before any forecasting)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()  # <-- blocks here until you close the window


# ---------------------------------------------------------------------------
# STEP 2: Fit a Prophet model
# ---------------------------------------------------------------------------
print("STEP 2: Fitting Prophet model... (no graph for this step, just math)")
model = Prophet(
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False,
    interval_width=0.80,
)
model.fit(df)
print("Model fit complete.")
print("-" * 60)


# ---------------------------------------------------------------------------
# STEP 3: Predict the future
# ---------------------------------------------------------------------------
HORIZON = 90
print(f"STEP 3: Predicting next {HORIZON} days...")

future = model.make_future_dataframe(periods=HORIZON)
forecast = model.predict(future)

print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(5))
print("-" * 60)


# ---- GRAPH 2: Actual vs Predicted, with confidence interval ----
print("Showing GRAPH 2: actual vs predicted + confidence interval. Close window to continue...")
plt.figure(figsize=(12, 6))
plt.plot(df["ds"], df["y"], "k.", label="Actual", alpha=0.5, markersize=3)
plt.plot(forecast["ds"], forecast["yhat"], label="Predicted", color="blue")
plt.fill_between(
    forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
    color="blue", alpha=0.2, label="Confidence Interval"
)
plt.legend()
plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()  # <-- blocks again


# ---- GRAPH 3: Trend / seasonality breakdown (Prophet's native plot) ----
print("Showing GRAPH 3: trend + weekly + yearly breakdown. Close window to continue...")
fig = model.plot_components(forecast)
plt.show()  # <-- blocks again


# ---------------------------------------------------------------------------
# STEP 4: Simple accuracy check (backtest) — no graph, just numbers
# ---------------------------------------------------------------------------
print("STEP 4: Quick backtest — train on all but last 90 days, test on those days...")

train = df.iloc[:-HORIZON]
test = df.iloc[-HORIZON:]

backtest_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
backtest_model.fit(train)

backtest_future = backtest_model.make_future_dataframe(periods=HORIZON)
backtest_forecast = backtest_model.predict(backtest_future)

pred = backtest_forecast.set_index("ds").loc[test["ds"], "yhat"].values
actual = test["y"].values

mae = np.mean(np.abs(pred - actual))
rmse = np.sqrt(np.mean((pred - actual) ** 2))

print(f"MAE  (average error size): {mae:.2f}")
print(f"RMSE (penalizes big misses more): {rmse:.2f}")
print("-" * 60)
print("DONE. All 3 graphs shown one by one.")