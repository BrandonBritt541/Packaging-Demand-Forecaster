"""
Packaging Demand Forecaster + Reorder Advisor
=============================================
A portfolio project for forecasting packaging SKU demand and advising reorders.

HOW TO RUN:
    streamlit run packaging_forecaster.py

The app opens in your browser. Phase 1 (data + chart) works right now.
Phases 2-4 are marked with TODO blocks for you to vibecode next.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

# ---------------------------------------------------------------------------
# SAMPLE DATA GENERATOR
# Creates realistic weekly demand for a few packaging SKUs, with a rising
# trend and yearly seasonality baked in -- so the forecast has real patterns
# to find. Replace this later by uploading your own CSV.
# ---------------------------------------------------------------------------
def generate_sample_data(weeks=156, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=weeks, freq="W")

    skus = {
        "SKU-1183 (Corrugated Box, Medium)": {"base": 4000, "trend": 8,  "season": 900},
        "SKU-2094 (Stretch Wrap Roll)":       {"base": 1500, "trend": 3,  "season": 250},
        "SKU-3376 (Poly Mailer, Large)":      {"base": 6000, "trend": 20, "season": 2200},  # holiday-heavy
    }

    frames = []
    for name, p in skus.items():
        t = np.arange(weeks)
        trend = p["base"] + p["trend"] * t
        # yearly seasonality (52-week cycle), peaking toward year-end for the mailer
        season = p["season"] * np.sin(2 * np.pi * (t % 52) / 52 + 1.5)
        noise = rng.normal(0, p["base"] * 0.05, weeks)
        qty = np.clip(trend + season + noise, 0, None).round()
        frames.append(pd.DataFrame({"date": dates, "sku": name, "quantity": qty}))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Packaging Demand Forecaster", layout="wide")
st.title("📦 Packaging Demand Forecaster + Reorder Advisor")

# --- Data source ---
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload demand CSV (date, sku, quantity)", type="csv")
    st.caption("No file? The app uses generated sample data.")

if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=["date"])
else:
    df = generate_sample_data()

sku_list = sorted(df["sku"].unique())
sku = st.sidebar.selectbox("SKU", sku_list)
sku_df = df[df["sku"] == sku].sort_values("date")

# ===========================================================================
# PHASE 1 -- Data + visualization  (WORKING)
# ===========================================================================
st.subheader("Historical Demand")

fig = go.Figure()
fig.add_trace(go.Scatter(x=sku_df["date"], y=sku_df["quantity"],
                         mode="lines", name="Actual demand"))
fig.update_layout(xaxis_title="Week", yaxis_title="Units", height=420)
st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Avg weekly demand", f"{sku_df['quantity'].mean():,.0f}")
c2.metric("Weeks of history", f"{len(sku_df)}")
c3.metric("Std deviation", f"{sku_df['quantity'].std():,.0f}")

# ===========================================================================
# PHASE 2 -- Forecast  (TODO: vibecode this)
# ---------------------------------------------------------------------------
# Prompt idea:
#   "Using StatsForecast with an AutoARIMA model, forecast the next 12 weeks
#    of demand for the selected SKU. StatsForecast wants columns named
#    unique_id, ds (date), and y (quantity). Plot the forecast as a new line
#    on the chart above with an 80% confidence interval band."
# What you'll learn: trend, seasonality, prediction intervals.
# ===========================================================================
st.subheader("Forecast")

# How many weeks ahead to predict (user-adjustable)
horizon = st.sidebar.slider("Forecast weeks ahead", min_value=4, max_value=26, value=12)

# StatsForecast expects columns named: unique_id, ds (date), y (value)
ts = sku_df.rename(columns={"date": "ds", "quantity": "y"}).copy()
ts["unique_id"] = sku

with st.spinner("Fitting AutoARIMA model... (first run takes ~10-20 sec)"):
    sf = StatsForecast(
        models=[AutoARIMA(season_length=52)],  # 52 = yearly seasonality on weekly data
        freq="W",
    )
    sf.fit(ts[["unique_id", "ds", "y"]])
    # level=[80, 95] returns 80% and 95% prediction intervals
    fc = sf.predict(h=horizon, level=[80, 95])

# Rebuild the chart: history + forecast line + shaded intervals
fig2 = go.Figure()

# historical actuals
fig2.add_trace(go.Scatter(x=ts["ds"], y=ts["y"],
                          mode="lines", name="Actual demand",
                          line=dict(color="#4C9BE8")))

# 95% interval band (widest, drawn first / faintest)
fig2.add_trace(go.Scatter(
    x=list(fc["ds"]) + list(fc["ds"][::-1]),
    y=list(fc["AutoARIMA-hi-95"]) + list(fc["AutoARIMA-lo-95"][::-1]),
    fill="toself", fillcolor="rgba(255,165,0,0.12)",
    line=dict(color="rgba(0,0,0,0)"), name="95% interval", showlegend=True))

# 80% interval band (narrower)
fig2.add_trace(go.Scatter(
    x=list(fc["ds"]) + list(fc["ds"][::-1]),
    y=list(fc["AutoARIMA-hi-80"]) + list(fc["AutoARIMA-lo-80"][::-1]),
    fill="toself", fillcolor="rgba(255,165,0,0.25)",
    line=dict(color="rgba(0,0,0,0)"), name="80% interval", showlegend=True))

# forecast central line
fig2.add_trace(go.Scatter(x=fc["ds"], y=fc["AutoARIMA"],
                          mode="lines", name="Forecast",
                          line=dict(color="orange", dash="dash")))

fig2.update_layout(xaxis_title="Week", yaxis_title="Units", height=420)
st.plotly_chart(fig2, use_container_width=True)

st.caption(f"Forecasting {horizon} weeks ahead with AutoARIMA. "
           "The shaded bands show the range demand is likely to fall within "
           "\u2014 wider bands mean more uncertainty.")

# ===========================================================================
# PHASE 3 -- Reorder advisor  (TODO: vibecode this)
# ---------------------------------------------------------------------------
# Inputs to add (sidebar): current stock on hand, supplier lead time (weeks),
#   desired service level (e.g. 95%).
# Concepts to implement:
#   - lead-time demand   = avg weekly demand * lead time
#   - safety stock       = z * demand_std * sqrt(lead_time)   (z=1.65 for 95%)
#   - reorder point      = lead-time demand + safety stock
# Output a plain sentence: "Reorder X units of <SKU> when stock hits Y."
# ===========================================================================
st.subheader("Reorder Recommendation")

# --- Inputs (sidebar) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Reorder settings")
current_stock = st.sidebar.number_input("Current stock on hand (units)",
                                        min_value=0, value=8000, step=500)
lead_time_weeks = st.sidebar.number_input("Supplier lead time (weeks)",
                                          min_value=1, max_value=26, value=4)
service_level = st.sidebar.slider("Service level (%)",
                                  min_value=80, max_value=99, value=95)

# z-score (safety factor) for the chosen service level.
# Higher service level -> higher z -> more safety stock -> fewer stockouts.
from scipy.stats import norm
z = norm.ppf(service_level / 100)

# Demand during lead time: sum the forecast over the lead-time weeks.
# This is the "use the forecast" choice -- the reorder point adapts to
# the trend instead of relying on a flat historical average.
lead_forecast = fc["AutoARIMA"].iloc[:lead_time_weeks].sum()

# Demand variability (std of historical weekly demand), scaled to lead time.
weekly_std = sku_df["quantity"].std()
safety_stock = z * weekly_std * np.sqrt(lead_time_weeks)

reorder_point = lead_forecast + safety_stock
recommended_order_qty = lead_forecast + safety_stock  # order enough to cover lead time + buffer

# --- Output ---
m1, m2, m3 = st.columns(3)
m1.metric("Lead-time demand (forecast)", f"{lead_forecast:,.0f}")
m2.metric("Safety stock", f"{safety_stock:,.0f}")
m3.metric("Reorder point", f"{reorder_point:,.0f}")

if current_stock <= reorder_point:
    st.error(
        f"**Reorder now.** Stock on hand ({current_stock:,.0f}) is at or below "
        f"the reorder point ({reorder_point:,.0f}). "
        f"Order about **{recommended_order_qty:,.0f} units** of {sku} to cover the "
        f"{lead_time_weeks}-week lead time at a {service_level}% service level."
    )
else:
    weeks_of_cover = (current_stock - reorder_point)
    st.success(
        f"**No reorder needed yet.** Stock on hand ({current_stock:,.0f}) is above "
        f"the reorder point ({reorder_point:,.0f}). "
        f"Reorder about **{recommended_order_qty:,.0f} units** once stock drops to "
        f"{reorder_point:,.0f}."
    )

st.caption(
    "Reorder point = forecasted demand over the lead time + safety stock. "
    "Safety stock = z \u00d7 weekly demand variability \u00d7 \u221a(lead time), "
    f"where z ({z:.2f}) comes from the {service_level}% service level."
)

# ===========================================================================
# PHASE 4 -- Backtest / validation  (TODO: vibecode this -- MOST IMPORTANT)
# ---------------------------------------------------------------------------
# Hold out the last ~12 weeks, forecast them, compare to what actually happened.
# Report an error metric (MAPE or RMSE) and plot forecast vs actual.
# This is what proves your forecast is trustworthy -- do not skip it.
# ===========================================================================
st.subheader("Backtest (accuracy check)")
st.info("Phase 4: hold out recent weeks, forecast them, report the error. This is the credibility step.")
