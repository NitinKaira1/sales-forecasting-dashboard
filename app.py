import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_components_plotly
import plotly.graph_objects as go

st.set_page_config(page_title="Sales & Demand Forecasting Dashboard", layout="wide")


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


@st.cache_resource(show_spinner=False)
def fit_model(df, weekly_seasonality, yearly_seasonality, interval_width):
    m = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    m.fit(df)
    return m


def compute_backtest_metrics(df, horizon_days, weekly_seasonality, yearly_seasonality):
    if len(df) <= horizon_days + 30:
        return None

    train = df.iloc[:-horizon_days]
    test = df.iloc[-horizon_days:]

    m = Prophet(
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=False,
    )
    m.fit(train)

    future = m.make_future_dataframe(periods=horizon_days)
    forecast = m.predict(future)
    pred = forecast.set_index("ds").loc[test["ds"], "yhat"].values
    actual = test["y"].values

    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    return {"mae": mae, "rmse": rmse, "n_days": horizon_days}


st.sidebar.header("1. Data Source")
data_source = st.sidebar.radio("Choose data source", ["Use sample data", "Upload my own CSV"])

df = None

if data_source == "Use sample data":
    df = generate_sample_sales_data()
    st.sidebar.success("Using generated sample sales data (2 years, daily)")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.write("Preview:")
        st.sidebar.dataframe(raw_df.head(), height=150)

        date_col = st.sidebar.selectbox("Which column is the date?", raw_df.columns)
        value_col = st.sidebar.selectbox(
            "Which column is the value to forecast?",
            [c for c in raw_df.columns if c != date_col],
        )

        df = raw_df[[date_col, value_col]].rename(columns={date_col: "ds", value_col: "y"})
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna().sort_values("ds").reset_index(drop=True)

        if df.empty:
            st.sidebar.error("Couldn't parse that date/value column combination. Check the CSV.")
            st.stop()
    else:
        st.info("Upload a CSV from the sidebar, or switch to sample data, to see the dashboard.")
        st.stop()

st.sidebar.header("2. Forecast Settings")
horizon = st.sidebar.slider("Days to forecast into the future", 7, 365, 90)
interval_width = st.sidebar.slider("Confidence interval width", 0.50, 0.95, 0.80, step=0.05)
weekly_seasonality = st.sidebar.checkbox("Weekly seasonality", value=True)
yearly_seasonality = st.sidebar.checkbox("Yearly seasonality", value=True)
run_backtest = st.sidebar.checkbox("Run accuracy backtest (MAE/RMSE)", value=True)

st.title("📈 Sales / Demand Forecasting Dashboard")
st.caption("Forecasting powered by Facebook/Meta Prophet")

st.subheader("Input Data")
col_a, col_b = st.columns([2, 1])
with col_a:
    st.dataframe(df.tail(10), use_container_width=True)
with col_b:
    st.metric("Total rows", len(df))
    st.metric("Date range", f"{df['ds'].min().date()} → {df['ds'].max().date()}")
    st.metric("Average value", f"{df['y'].mean():.1f}")

with st.spinner("Fitting Prophet model..."):
    model = fit_model(df, weekly_seasonality, yearly_seasonality, interval_width)

future = model.make_future_dataframe(periods=horizon)
forecast = model.predict(future)

st.subheader("Actual vs Predicted (with confidence interval)")
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat_upper"],
    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"
))
fig1.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat_lower"],
    mode="lines", line=dict(width=0), fill="tonexty",
    fillcolor="rgba(99,110,250,0.2)", name=f"{int(interval_width*100)}% Confidence Interval",
    hoverinfo="skip"
))
fig1.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode="lines", name="Predicted", line=dict(color="royalblue")
))
fig1.add_trace(go.Scatter(
    x=df["ds"], y=df["y"],
    mode="markers", name="Actual", marker=dict(color="black", size=4, opacity=0.6)
))

fig1.update_layout(
    height=480, xaxis_title="Date", yaxis_title="Value",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(t=40),
)
st.plotly_chart(fig1, use_container_width=True)

if run_backtest:
    metrics = compute_backtest_metrics(df, min(horizon, max(7, len(df) // 5)), weekly_seasonality, yearly_seasonality)
    if metrics:
        st.subheader("Backtest Accuracy")
        st.caption(f"Model trained on all data except the last {metrics['n_days']} days, then evaluated on those held-out days.")
        c1, c2 = st.columns(2)
        c1.metric("MAE (Mean Absolute Error)", f"{metrics['mae']:.2f}")
        c2.metric("RMSE (Root Mean Squared Error)", f"{metrics['rmse']:.2f}")
    else:
        st.caption("Not enough historical data to run a meaningful backtest.")

st.subheader(f"Forecast — Next {horizon} Days")
forecast_display = (
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    .tail(horizon)
    .rename(columns={
        "ds": "Date", "yhat": "Predicted",
        "yhat_lower": "Lower Bound", "yhat_upper": "Upper Bound"
    })
)
st.dataframe(forecast_display, use_container_width=True)

csv_bytes = forecast_display.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download forecast as CSV", csv_bytes, "forecast.csv", "text/csv")

st.subheader("Trend & Seasonality Breakdown")
fig2 = plot_components_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("ℹ️ What am I looking at? (Trend / Seasonality explained)"):
    st.markdown("""
    - **Trend**: the long-term direction of the data with random noise removed —
      is the underlying quantity generally growing, shrinking, or flat over time,
      and where did that direction change (Prophet's "changepoints")?
    - **Weekly seasonality**: the repeating pattern within a week — e.g. do
      weekends consistently run higher or lower than weekdays?
    - **Yearly seasonality**: the repeating pattern across a year — e.g. a
      holiday-season bump, a summer slump, etc.
    - **Confidence interval (shaded band)**: the range Prophet considers
      plausible for the actual future value, not just a single number. It
      widens the further into the future you forecast, because uncertainty
      compounds — this is expected and correct behavior, not a bug.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Prophet + Streamlit · Sales/Demand Forecasting Dashboard")