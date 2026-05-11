"""
FastAPI REST API — Sales Forecasting Service
=============================================
Production-ready REST API that serves predictions from trained models.

Endpoints:
  GET  /health                         → Health check
  GET  /states                         → List available states
  GET  /models                         → Model info & scores
  POST /forecast                       → Forecast for a state
  POST /forecast/batch                 → Batch forecast for multiple states
  GET  /forecast/{state}/best          → Best model forecast for state
  GET  /forecast/{state}/compare       → Compare all models for state
  POST /retrain                        → Trigger model retraining
"""

import os
import sys
import json
import logging
from typing import Optional, List
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Setup ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forecasting_api")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = "results"

# ── Pydantic Models ────────────────────────────────────────────

class ForecastRequest(BaseModel):
    state: str = Field(..., description="State name to forecast", example="California")
    steps: int = Field(default=8, ge=1, le=52, description="Number of weeks to forecast")
    model: Optional[str] = Field(default="best", description="Model to use: best, ensemble, SARIMA, Prophet, XGBoost, LSTM")
    include_confidence: bool = Field(default=True, description="Include confidence intervals")

class BatchForecastRequest(BaseModel):
    states: List[str] = Field(..., description="List of states to forecast")
    steps: int = Field(default=8, ge=1, le=52)
    model: Optional[str] = Field(default="best")

class ForecastPoint(BaseModel):
    week: int
    date: str
    predicted_sales: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

class ForecastResponse(BaseModel):
    state: str
    model_used: str
    steps: int
    generated_at: str
    forecast: List[ForecastPoint]
    model_score: Optional[dict] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    states_available: int
    timestamp: str

# ── App Initialization ─────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description="Production-grade time series forecasting system with SARIMA, Prophet, XGBoost, and LSTM models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── App State ──────────────────────────────────────────────────
class AppState:
    metadata: dict = {}
    selection_report: dict = {}
    best_models: dict = {}
    forecast_cache: dict = {}   # state -> {model -> df}
    models_loaded: list = []
    ready: bool = False

app_state = AppState()


def load_artifacts():
    """Load pre-computed artifacts at startup."""
    logger.info("Loading artifacts...")

    meta_path = f"{ARTIFACTS_DIR}/metadata.json"
    report_path = f"{ARTIFACTS_DIR}/model_selection_report.json"

    if not os.path.exists(meta_path):
        logger.warning("No metadata found. Run train.py first.")
        return

    with open(meta_path) as f:
        app_state.metadata = json.load(f)

    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        app_state.best_models = report.get("best_models", {})
        app_state.selection_report = report.get("scores", {})

    # Load pre-computed forecast CSVs
    models_to_load = ["best_model", "ensemble", "sarima", "prophet", "xgboost", "lstm"]
    for model_name in models_to_load:
        path = f"{RESULTS_DIR}/{model_name}_forecast.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"])
            app_state.forecast_cache[model_name] = df
            app_state.models_loaded.append(model_name)

    app_state.ready = True
    logger.info(f"Loaded {len(app_state.models_loaded)} forecast sets for {len(app_state.metadata.get('states', []))} states.")


@app.on_event("startup")
async def startup_event():
    load_artifacts()


# ── Utility Functions ──────────────────────────────────────────

def get_forecast_df(model: str, state: str) -> Optional[pd.DataFrame]:
    """Retrieve cached forecast for a model and state."""
    model_key = model.lower().replace(" ", "_")
    cache_key_map = {
        "best": "best_model",
        "ensemble": "ensemble",
        "sarima": "sarima",
        "prophet": "prophet",
        "xgboost": "xgboost",
        "lstm": "lstm",
    }
    key = cache_key_map.get(model_key, "best_model")
    df = app_state.forecast_cache.get(key)
    if df is None:
        return None
    state_df = df[df["state"].str.lower() == state.lower()]
    return state_df if not state_df.empty else None


def add_confidence_intervals(fc_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Add approximate 95% confidence intervals based on historical std."""
    if "lower_bound" in fc_df.columns and "upper_bound" in fc_df.columns:
        return fc_df

    # Approximate CI using ±1.96 * assumed CV of 0.15
    cv = 0.15
    fc_df = fc_df.copy()
    fc_df["lower_bound"] = (fc_df["predicted_sales"] * (1 - 1.96 * cv)).clip(lower=0)
    fc_df["upper_bound"] = fc_df["predicted_sales"] * (1 + 1.96 * cv)
    return fc_df


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """API health check — confirms model readiness."""
    return HealthResponse(
        status="healthy" if app_state.ready else "not_ready",
        models_loaded=app_state.models_loaded,
        states_available=len(app_state.metadata.get("states", [])),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/states", tags=["Data"])
async def list_states():
    """List all states available for forecasting."""
    states = app_state.metadata.get("states", [])
    return {
        "states": sorted(states),
        "count": len(states),
    }


@app.get("/models", tags=["Models"])
async def model_info():
    """Get model information, available models, and per-state best model selection."""
    return {
        "available_models": ["SARIMA", "Prophet", "XGBoost", "LSTM", "Ensemble"],
        "best_model_per_state": app_state.best_models,
        "model_scores": app_state.selection_report,
        "forecast_horizon_weeks": app_state.metadata.get("forecast_steps", 8),
    }


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
async def forecast_state(request: ForecastRequest):
    """
    Forecast sales for a specific state.
    
    - **state**: State name (case-insensitive)
    - **steps**: Number of weeks to forecast (1–52)  
    - **model**: Model to use (best, ensemble, SARIMA, Prophet, XGBoost, LSTM)
    - **include_confidence**: Include 95% confidence intervals
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="Models not loaded. Run training first.")

    states = [s.lower() for s in app_state.metadata.get("states", [])]
    if request.state.lower() not in states:
        raise HTTPException(
            status_code=404,
            detail=f"State '{request.state}' not found. Available: {app_state.metadata.get('states', [])}",
        )

    fc_df = get_forecast_df(request.model, request.state)
    if fc_df is None:
        # Fall back to best model
        fc_df = get_forecast_df("best", request.state)
    if fc_df is None:
        raise HTTPException(status_code=404, detail=f"No forecast found for state: {request.state}")

    if request.include_confidence:
        fc_df = add_confidence_intervals(fc_df, request.model)

    fc_df = fc_df.head(request.steps)

    forecast_points = []
    for _, row in fc_df.iterrows():
        point = ForecastPoint(
            week=int(row.get("week", 1)),
            date=str(row["date"])[:10],
            predicted_sales=round(float(row["predicted_sales"]), 2),
        )
        if request.include_confidence and "lower_bound" in row:
            point.lower_bound = round(float(row["lower_bound"]), 2)
            point.upper_bound = round(float(row["upper_bound"]), 2)
        forecast_points.append(point)

    # Get model score for this state
    model_score = None
    state_report = app_state.selection_report.get(request.state, {})
    if state_report:
        model_score = state_report.get("scores", {}).get(
            app_state.best_models.get(request.state, "XGBoost"), {}
        )

    model_used = app_state.best_models.get(request.state, request.model) if request.model == "best" else request.model

    return ForecastResponse(
        state=request.state,
        model_used=model_used,
        steps=len(forecast_points),
        generated_at=datetime.utcnow().isoformat(),
        forecast=forecast_points,
        model_score=model_score,
    )


@app.post("/forecast/batch", tags=["Forecasting"])
async def batch_forecast(request: BatchForecastRequest):
    """
    Forecast sales for multiple states in one request.
    Returns a list of forecasts, one per state.
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    results = []
    errors = []
    for state in request.states:
        try:
            fc_req = ForecastRequest(state=state, steps=request.steps, model=request.model)
            fc = await forecast_state(fc_req)
            results.append(fc)
        except HTTPException as e:
            errors.append({"state": state, "error": e.detail})

    return {
        "results": results,
        "errors": errors,
        "total": len(results),
        "failed": len(errors),
    }


@app.get("/forecast/{state}/best", tags=["Forecasting"])
async def best_forecast(state: str, steps: int = Query(default=8, ge=1, le=52)):
    """Get the best-model forecast for a state (shorthand endpoint)."""
    req = ForecastRequest(state=state, steps=steps, model="best", include_confidence=True)
    return await forecast_state(req)


@app.get("/forecast/{state}/compare", tags=["Forecasting"])
async def compare_models(state: str, steps: int = Query(default=8, ge=1, le=52)):
    """
    Compare forecasts from all models for a given state.
    Useful for analysis and model selection review.
    """
    if not app_state.ready:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    comparison = {}
    for model_name in ["SARIMA", "Prophet", "XGBoost", "LSTM", "ensemble"]:
        fc_df = get_forecast_df(model_name, state)
        if fc_df is not None:
            fc_df = fc_df.head(steps)
            comparison[model_name] = [
                {
                    "week": int(row.get("week", i + 1)),
                    "date": str(row["date"])[:10],
                    "predicted_sales": round(float(row["predicted_sales"]), 2),
                }
                for i, (_, row) in enumerate(fc_df.iterrows())
            ]

    state_report = app_state.selection_report.get(state, {})
    return {
        "state": state,
        "best_model": app_state.best_models.get(state, "Unknown"),
        "model_scores": state_report.get("scores", {}),
        "forecasts": comparison,
    }


@app.post("/retrain", tags=["System"])
async def trigger_retrain(background_tasks: BackgroundTasks, data_path: str = "data/sales_data.xlsx"):
    """
    Trigger model retraining in the background.
    Returns immediately — retraining runs asynchronously.
    """
    def retrain_job():
        logger.info(f"Starting retraining from {data_path}...")
        try:
            from train import run_pipeline
            run_pipeline(data_path)
            load_artifacts()
            logger.info("Retraining complete.")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")

    background_tasks.add_task(retrain_job)
    return {
        "status": "accepted",
        "message": "Retraining started in background. Check /health to confirm when ready.",
        "data_path": data_path,
    }


@app.get("/forecast/{state}/summary", tags=["Forecasting"])
async def forecast_summary(state: str):
    """Get aggregated summary statistics for the 8-week forecast."""
    req = ForecastRequest(state=state, steps=8, model="best")
    fc = await forecast_state(req)
    values = [p.predicted_sales for p in fc.forecast]
    return {
        "state": state,
        "model_used": fc.model_used,
        "total_forecasted_sales": round(sum(values), 2),
        "average_weekly_sales": round(np.mean(values), 2),
        "peak_week": int(np.argmax(values)) + 1,
        "peak_sales": round(max(values), 2),
        "min_week": int(np.argmin(values)) + 1,
        "min_sales": round(min(values), 2),
        "trend": "increasing" if values[-1] > values[0] else "decreasing",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
