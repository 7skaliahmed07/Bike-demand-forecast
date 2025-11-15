# src/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Bike Demand Predictor", version="1.0")

# Load model
model_path = "../models/xgboost_best.pkl"
model = joblib.load(model_path)

class InputData(BaseModel):
    year: int
    month: int
    day: int
    hour: int
    temp: float
    hum: float
    windspeed: float
    weathersit: int
    holiday: int = 0
    workingday: int = 1
    weekday: int

@app.get("/")
def home():
    return {"message": "Bike Demand API – use POST /predict"}

@app.post("/predict")
def predict(data: InputData):
    # Reconstruct feature vector (same order as training)
    features = [
        data.year, data.month, data.day, data.hour,
        data.temp, data.hum, data.windspeed,
        data.weathersit, data.holiday, data.workingday, data.weekday,
        int(data.hour in [7,8,9]),           # is_morning_rush
        int(data.hour in [16,17,18,19]),     # is_evening_rush
        data.temp * data.hum                 # temp_hum
    ]
    pred = model.predict([features])[0]
    return {"predicted_hourly_demand": round(float(pred))}