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
st.info("Phase 2: add an AutoARIMA forecast here. See the prompt in the code comments.")

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
st.info("Phase 3: compute safety stock and reorder point. See comments for the formulas.")

# ===========================================================================
# PHASE 4 -- Backtest / validation  (TODO: vibecode this -- MOST IMPORTANT)
# ---------------------------------------------------------------------------
# Hold out the last ~12 weeks, forecast them, compare to what actually happened.
# Report an error metric (MAPE or RMSE) and plot forecast vs actual.
# This is what proves your forecast is trustworthy -- do not skip it.
# ===========================================================================
st.subheader("Backtest (accuracy check)")
st.info("Phase 4: hold out recent weeks, forecast them, report the error. This is the credibility step.")
