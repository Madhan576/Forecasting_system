"""
Main Training Pipeline
=======================
Orchestrates the full forecasting workflow:
  1. Data loading & feature engineering
  2. Train SARIMA, Prophet, XGBoost, LSTM
  3. Evaluate all models on validation set
  4. Select best model per state
  5. Save all artifacts
"""

from models.model_selector import ModelSelector
from models.lstm_model import LSTMForecaster
from models.xgboost_model import XGBoostForecaster
from models.prophet_model import ProphetForecaster
from models.sarima_model import SARIMAForecaster
from data.preprocessor import DataPreprocessor
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


ARTIFACTS_DIR = "artifacts"
RESULTS_DIR = "results"


def run_pipeline(data_path: str, val_weeks: int = 8, forecast_steps: int = 8):
    print("\n" + "=" * 60)
    print("TIME SERIES FORECASTING SYSTEM — TRAINING PIPELINE")
    print("=" * 60 + "\n")

    # ── Step 1: Preprocessing ──────────────────────────────────
    print("STEP 1: Data Preprocessing & Feature Engineering")
    preprocessor = DataPreprocessor()
    full_df, train_df, val_df = preprocessor.preprocess_pipeline(
        data_path, val_weeks=val_weeks)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    try:
        full_df.to_parquet(
            f"{ARTIFACTS_DIR}/processed_data.parquet", index=False)
        train_df.to_parquet(f"{ARTIFACTS_DIR}/train_data.parquet", index=False)
        val_df.to_parquet(f"{ARTIFACTS_DIR}/val_data.parquet", index=False)
        print(f"  Saved processed data to {ARTIFACTS_DIR}/ (parquet)")
    except ImportError:
        full_df.to_csv(f"{ARTIFACTS_DIR}/processed_data.csv", index=False)
        train_df.to_csv(f"{ARTIFACTS_DIR}/train_data.csv", index=False)
        val_df.to_csv(f"{ARTIFACTS_DIR}/val_data.csv", index=False)
        print(f"  Saved processed data to {ARTIFACTS_DIR}/ (csv fallback)")

    states = full_df["state"].unique().tolist()
    print(f"  States: {states}\n")

    # ── Step 2: Train Models ───────────────────────────────────
    all_scores = {}

    # SARIMA
    print("STEP 2a: Training SARIMA")
    sarima = SARIMAForecaster()
    sarima.fit(train_df)
    sarima_scores = sarima.evaluate(val_df)
    sarima.save(ARTIFACTS_DIR)
    all_scores["SARIMA"] = sarima_scores

    # Prophet
    print("\nSTEP 2b: Training Prophet")
    try:
        prophet = ProphetForecaster()
        prophet.fit(train_df)
        prophet_scores = prophet.evaluate(val_df)
        prophet.save(ARTIFACTS_DIR)
        all_scores["Prophet"] = prophet_scores
    except Exception as e:
        print(f"  [WARNING] Prophet failed: {e}")
        prophet = None
        all_scores["Prophet"] = {}

    # XGBoost
    print("\nSTEP 2c: Training XGBoost")
    xgb_model = XGBoostForecaster()
    xgb_model.fit(train_df, val_df)
    xgb_scores = xgb_model.evaluate(train_df, val_df)
    xgb_model.save(ARTIFACTS_DIR)
    all_scores["XGBoost"] = xgb_scores

    # LSTM
    print("\nSTEP 2d: Training LSTM")
    lstm = LSTMForecaster(epochs=50)
    lstm.fit(train_df, val_df)
    lstm_scores = lstm.evaluate(train_df, val_df)
    lstm.save(ARTIFACTS_DIR)
    all_scores["LSTM"] = lstm_scores

    # ── Step 3: Model Selection ────────────────────────────────
    print("\nSTEP 3: Model Selection")
    selector = ModelSelector()
    selector.collect_scores(
        sarima_scores=all_scores["SARIMA"],
        prophet_scores=all_scores["Prophet"],
        xgboost_scores=all_scores["XGBoost"],
        lstm_scores=all_scores["LSTM"],
    )
    selector.select_best(metric="rmse")
    selector.compute_ensemble_weights(metric="rmse")
    selector.print_leaderboard()
    selector.save_report(ARTIFACTS_DIR)

    # ── Step 4: Generate Forecasts ─────────────────────────────
    print("STEP 4: Generating 8-Week Forecasts")

    forecasts = {}

    # SARIMA forecasts
    sarima_fc = sarima.predict_all(states=states, steps=forecast_steps)
    forecasts["SARIMA"] = sarima_fc

    # Prophet forecasts
    if prophet:
        prophet_fc = prophet.predict_all(states=states, steps=forecast_steps)
        forecasts["Prophet"] = prophet_fc

    # XGBoost forecasts
    xgb_fc = xgb_model.predict_all(
        train_df=full_df, states=states, steps=forecast_steps)
    forecasts["XGBoost"] = xgb_fc

    # LSTM forecasts
    lstm_fc = lstm.predict_all(
        train_df=full_df, states=states, steps=forecast_steps)
    forecasts["LSTM"] = lstm_fc

    # Best-model combined forecast
    best_fc = selector.combine_forecasts(
        forecasts, states=states, steps=forecast_steps, strategy="best")
    # Ensemble forecast
    ensemble_fc = selector.combine_forecasts(
        forecasts, states=states, steps=forecast_steps, strategy="ensemble")

    # Save all forecasts
    best_fc.to_csv(f"{RESULTS_DIR}/best_model_forecast.csv", index=False)
    ensemble_fc.to_csv(f"{RESULTS_DIR}/ensemble_forecast.csv", index=False)
    for model_name, fc_df in forecasts.items():
        fc_df.to_csv(
            f"{RESULTS_DIR}/{model_name.lower()}_forecast.csv", index=False)

    print(f"\n  Forecasts saved to {RESULTS_DIR}/")

    # ── Step 5: Summary ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  States forecasted: {len(states)}")
    print(f"  Forecast horizon: {forecast_steps} weeks")
    print(f"  Artifacts: {ARTIFACTS_DIR}/")
    print(f"  Results:   {RESULTS_DIR}/")
    print("=" * 60 + "\n")

    # Save metadata for API
    metadata = {
        "states": states,
        "forecast_steps": forecast_steps,
        "val_weeks": val_weeks,
        "models_trained": list(all_scores.keys()),
        "best_models": selector.best_models,
    }
    with open(f"{ARTIFACTS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "states": states,
        "selector": selector,
        "forecasts": forecasts,
        "best_forecast": best_fc,
        "ensemble_forecast": ensemble_fc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset (xlsx or csv)")
    parser.add_argument("--val_weeks", type=int, default=8,
                        help="Weeks for validation")
    parser.add_argument("--forecast", type=int, default=8,
                        help="Weeks to forecast")
    args = parser.parse_args()

    run_pipeline(args.data, val_weeks=args.val_weeks,
                 forecast_steps=args.forecast)
