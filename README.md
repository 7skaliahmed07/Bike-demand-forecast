# Hourly Bike Demand Forecasting: A Scalable ML Pipeline

Predicts **next-hour bike rentals** using the UCI Bike Sharing Dataset (Washington, DC, 2011–2012). Demonstrates **end-to-end ML research**: data engineering, time-aware modeling, experiment tracking, and real-time inference.

> **Key Insight**: Feature engineering (rush-hour flags, temp-humidity interaction) + XGBoost reduces RMSE by **13%** vs Random Forest.

---

## Research Questions & Findings

| Question | Finding |
|--------|--------|
| **What drives demand?** | Peaks at **8–9 AM** and **5–7 TAC** on weekdays. `hour`, `workingday`, and `temp` dominate. |
| **Weather impact?** | `temp` ↑ → demand ↑; `weathersit=3` (rain) → **60% drop**. |
| **Best model?** | **XGBoost**: RMSE **63.99**, R² **0.849** (vs RF: 73.33, 0.801). |
| **Real-time prediction?** | FastAPI `/predict` endpoint returns result in **<50ms**. |

---

## Model Performance (Time-Aware Test: Last ~30 Days)

| Model         | RMSE  | R²    |
|---------------|-------|-------|
| Random Forest | 73.33 | 0.801 |
| **XGBoost**   | **63.99** | **0.849** |

> **+13% accuracy gain** via engineered features and gradient boosting.

---

## Methodology

- **Dataset**: 17,379 hourly records (`cnt` = total rentals)
- **Train/Test Split**: Time-aware — **last 718 rows (~30 days)** as holdout
- **Features**: 19 total (year, month, hour, weather, rush-hour flags, `temp×hum`)
- **Models**: Random Forest, XGBoost
- **Tracking**: MLflow (metrics, models, runs)
- **Deployment**: FastAPI with saved XGBoost model

---

## API Usage

```bash
uvicorn src.app:app --reload