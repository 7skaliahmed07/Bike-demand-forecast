# src/train.py
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib

# 1. Load engineered data
df = pd.read_csv("data/processed/engineered_bike_data.csv")
print("Loaded engineered data:", df.shape)

# 2. Restore date
df['dteday'] = pd.to_datetime(df['dteday'])

# 3. Time-aware split: last 30 days = test
split_date = df['dteday'].max() - pd.Timedelta(days=30)
train = df[df['dteday'] <= split_date]
test = df[df['dteday'] > split_date]

print(f"Train: {train.shape}, Test: {test.shape}")

# 4. Features & target
X_train = train.drop(['cnt', 'dteday'], axis=1)
y_train = train['cnt']
X_test = test.drop(['cnt', 'dteday'], axis=1)
y_test = test['cnt']

# 5. MLflow
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("bike_demand_forecast")

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        if "RandomForest" in name:
            mlflow.sklearn.log_model(model, "model")
        else:
            mlflow.xgboost.log_model(model, "model")
        
        print(f"{name} - RMSE: {rmse:.2f}, R²: {r2:.3f}")

        # Save best model
        if name == "XGBoost":
            os.makedirs("../models", exist_ok=True)
            joblib.dump(model, "../models/xgboost_best.pkl")
            print("XGBoost saved to models/xgboost_best.pkl")