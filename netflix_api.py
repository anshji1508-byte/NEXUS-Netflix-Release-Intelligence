"""FastAPI inference service for the versioned Joblib forecaster."""
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from netflix_ml import load_bundle, future_forecast

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "netflix_forecaster.joblib"
app = FastAPI(title="NEXUS Netflix Forecast API", version="1.0.0", description="Production-style inference API for Netflix release trend forecasts.")

class ForecastRequest(BaseModel):
    horizon: int = Field(default=12, ge=1, le=36)

def get_bundle():
    if not ARTIFACT.exists():
        raise HTTPException(status_code=503, detail="Model artifact unavailable. Run train_netflix_model.py first.")
    return load_bundle(ARTIFACT)

@app.get("/health")
def health():
    return {"status": "ok" if ARTIFACT.exists() else "degraded", "artifact": str(ARTIFACT), "artifact_exists": ARTIFACT.exists()}

@app.get("/metadata")
def metadata():
    bundle = get_bundle()
    return {"model": "HistGradientBoostingRegressor", "artifact": ARTIFACT.name, "features": bundle["features"], "metrics": bundle["metrics"]}

@app.get("/forecast")
def forecast(horizon: int = 12):
    if horizon < 1 or horizon > 36: raise HTTPException(status_code=422, detail="horizon must be between 1 and 36")
    bundle = get_bundle()
    return {"horizon": horizon, "forecast": future_forecast(bundle, horizon), "model_metrics": bundle["metrics"]}

@app.post("/forecast")
def forecast_post(request: ForecastRequest):
    return forecast(request.horizon)
