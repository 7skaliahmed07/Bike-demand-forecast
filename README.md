# Scalable ML Pipeline for Hourly Bike Demand Forecasting

Predicts **next-hour bike rentals** using time, weather, and engineered features from the **UCI Bike Sharing Dataset** (Washington, DC, 2011–2012).

> **Research-ready**: clean code, MLflow tracking, time-aware split, FastAPI deployment.

---

## Research Questions & Answers

| Question | Answer |
|--------|--------|
| **What drives demand?** | Peaks at **8–9 AM** and **5–7 PM** on weekdays. `hour`, `workingday`, and `temp` are top predictors. |
| **How does weather affect usage?** | Higher `temp` → ↑ demand. Rain (`weathersit=3`) → ↓ 60% rentals. |
| **Which model performs best?** | **XGBoost** (RMSE **63.99**) beats Random Forest (73.33). |
| **Can it run in real time?** | Yes — `POST /predict` returns demand in **<50ms**. |

---

## Model Results (Your Run)

| Model         | RMSE  | R²    |
|---------------|-------|-------|
| Random Forest | 73.33 | 0.801 |
| **XGBoost**   | **63.99** | **0.849** |

> **+13% improvement** with XGBoost. Feature engineering (rush-hour flags, temp-humidity interaction) reduced error vs baseline.

---

## Data & Split

- **Dataset**: `hour.csv` (17,379 rows)
- **Train**: 16,661 rows (first ~22 months)
- **Test**: 718 rows (**last ~30 days**, time-aware split)
- **Target**: `cnt` (hourly rentals)
- **Features**: 19 engineered (year, month, hour, temp, hum, windspeed, weathersit, weekday, holiday, workingday, rush flags, temp×hum)

