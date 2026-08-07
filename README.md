# 📦 Packaging Demand Forecaster + Reorder Advisor

A tool that forecasts demand for packaging SKUs and recommends when and how much
to reorder — built to bring AI/ML forecasting to packaging distribution and
supply chain workflows.

## What it does

- **Visualizes** historical weekly demand for packaging SKUs
- **Forecasts** future demand using time-series models (AutoARIMA)
- **Advises reorders** by calculating safety stock and reorder points from
  demand variability and supplier lead time
- **Validates** its own accuracy with a backtest against held-out history

Upload your own demand CSV (`date`, `sku`, `quantity`), or explore with the
built-in sample data.

## Run it

**Live app:** _(add your Streamlit Cloud URL here once deployed)_

**Locally:**
```
pip install -r requirements.txt
streamlit run packaging_forecaster.py
```

## Why this project

Accurate demand forecasting keeps inventory lean without stocking out —
one of the highest-ROI applications of ML in supply chain. This project pairs
forecasting with the inventory math (safety stock, reorder points) that turns a
prediction into an actual purchasing decision.

## Tech

Python · Streamlit · StatsForecast · pandas · Plotly

## Roadmap

- [x] Data ingestion + visualization
- [ ] AutoARIMA demand forecast with confidence intervals
- [ ] Reorder point + safety stock advisor
- [ ] Backtest / accuracy reporting
