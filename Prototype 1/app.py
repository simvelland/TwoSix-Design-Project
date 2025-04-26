from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd

model = joblib.load('trained_gbr_model.pkl')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/")
def predict(data: dict):
    try:
        features = np.array([[
            data["Solar_Irradiance"], 
            data["Temperature_2m"],
            data["Relative_Humidity_2m"],
            data["Wind_Speed_2M"],
            data["Surface_Pressure"],
            data["Cloud_Amount"]
        ]])

        prediction = model.predict(features)[0]

        return {"predicted_daily_solar_power": prediction}

    except Exception as e:
        return {"error": str(e)}