# main.py
# Run this file using: uvicorn main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import re
import pandas as pd
import numpy as np
import pickle
from datetime import timedelta

app = FastAPI(title="Hospify ML API")

# Allow the React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR  = "data"
MODEL_DIR = "models"
_model_cache = {}

PROPERTY_CFG = {
    "Data 1": {
        "history_csv":  os.path.join(DATA_DIR, "history_1.csv"),
        "model_key":    "data_1",
        "model_path":   os.path.join(MODEL_DIR, "data_1.pkl"),
        "fetch_limit":  400,
        "engine_type":  "ensemble",
        "transform":    "none",
        "jan_correction": False,
        "inventory":    126,
    },
    "Data 2": {
        "history_csv":  os.path.join(DATA_DIR, "history_2.csv"),
        "model_key":    "data_2",
        "model_path":   os.path.join(MODEL_DIR, "data_2.pkl"),
        "fetch_limit":  60,
        "engine_type":  "direct",
        "transform":    "log1p", 
        "jan_correction": True,  
        "inventory":    100,
    },
    "Data 3": {
        "history_csv":  os.path.join(DATA_DIR, "history_3.csv"),
        "model_key":    "data_3",
        "model_path":   os.path.join(MODEL_DIR, "data_3.pkl"),
        "fetch_limit":  60,
        "engine_type":  "direct",
        "transform":    "cube",  
        "jan_correction": False,
        "inventory":    185,
    },
}

def _get_model(key: str, path: str):
    if key not in _model_cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found: {path}")
        with open(path, "rb") as f:
            _model_cache[key] = pickle.load(f)
    return _model_cache[key]

# --- BUG FIX: Feature Name Extraction ---
def _model_feature_names(model) -> list:
    try:
        # FIXED: Removed the parentheses. It's a property, not a method.
        return list(model.feature_name_)
    except AttributeError:
        try:
            return list(model.booster_.feature_name())
        except AttributeError:
            return []

def _apply_inverse_transform(raw: float, transform: str) -> float:
    if transform == "log1p":
        return float(np.clip(np.expm1(raw), 0, 100))
    elif transform == "cube":
        return float(np.clip(np.cbrt(raw), 0, 100))
    else:
        return float(np.clip(raw, 0, 100))

# Reduced for brevity: Assuming _forecast_direct and _forecast_ensemble logic 
# remains the same as your engine.py, but returning dictionaries instead of DataFrames.

@app.get("/api/forecast")
def get_forecast(property: str, horizon: int = 30):
    if property not in PROPERTY_CFG:
        raise HTTPException(status_code=400, detail="Property not found")
        
    cfg = PROPERTY_CFG[property]
    
    try:
        # Note: To fully connect this, insert your full _forecast_direct 
        # and _forecast_ensemble functions here from your engine.py.
        # For the API endpoint, you return JSON:
        
        # Example Response Structure expected by React:
        # return {
        #     "property": property,
        #     "horizon": horizon,
        #     "forecast": df_fc.to_dict(orient="records") 
        # }
        
        return {"status": "Backend Connected. Insert engine logic here to return JSON."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))