"""
Facebook Prophet Model for State-Level Sales Forecasting
=========================================================
Handles trend, seasonality (weekly/yearly), and holidays automatically.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARNING] Prophet not installed. Run: pip install prophet")


class ProphetForecaster:
    """Prophet forecaster trained per state."""

    def __init__(self, country_holidays: str = "US"):
        self.country_holidays = country_holidays
        self.models = {}
        self.model_name = "Prophet"

    def _build_prophet_df(self, state_df: pd.DataFrame) -> pd.DataFrame:
        """Prophet requires columns: ds (datetime) and y (target)."""
        return state_df[["date", "sales"]].rename(columns={"date": "ds", "sales": "y"})

    def fit(self, train_df: pd.DataFrame):
        """Fit a Prophet model per state."""
        if not PROPHET_AVAILABLE:
            raise ImportError("Install prophet: pip install prophet")

        states = train_df["state"].unique()
        print(f"[Prophet] Fitting {len(states)} state models...")

        for state in states:
            state_df = (
                train_df[train_df["state"] == state]
                .sort_values("date")
                .reset_index(drop=True)
            )
            prophet_df = self._build_prophet_df(state_df)

            try:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",  # Better for sales data
                    changepoint_prior_scale=0.15,       # Controls trend flexibility
                    seasonality_prior_scale=10.0,
                    interval_width=0.95,
                )
                # Add US country holidays
                model.add_country_holidays(country_name=self.country_holidays)
                model.fit(prophet_df)
                self.models[state] = model
                print(f"  [Prophet] {state}: fitted on {len(prophet_df)} weeks.")
            except Exception as e:
                print(f"  [Prophet] {state}: failed — {e}")

        print(f"[Prophet] Fitted {len(self.models)} / {len(states)} models.")
        return self

    def predict(self, state: str, steps: int = 8) -> pd.DataFrame:
        """Forecast next N weeks for a given state."""
        if state not in self.models:
            raise ValueError(f"No Prophet model for state: {state}")
        model = self.models[state]
        future = model.make_future_dataframe(periods=steps, freq="W")
        forecast = model.predict(future)
        future_only = forecast.tail(steps)[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        future_only["yhat"] = future_only["yhat"].clip(lower=0)
        future_only["yhat_lower"] = future_only["yhat_lower"].clip(lower=0)
        return future_only

    def predict_all(self, states: list = None, steps: int = 8) -> pd.DataFrame:
        """Forecast for all (or specified) states."""
        states = states or list(self.models.keys())
        records = []
        for state in states:
            if state not in self.models:
                continue
            fc = self.predict(state, steps)
            for i, row in enumerate(fc.itertuples()):
                records.append({
                    "state": state,
                    "week": i + 1,
                    "date": row.ds,
                    "predicted_sales": round(float(row.yhat), 2),
                    "lower_bound": round(float(row.yhat_lower), 2),
                    "upper_bound": round(float(row.yhat_upper), 2),
                    "model": self.model_name,
                })
        return pd.DataFrame(records)

    def evaluate(self, val_df: pd.DataFrame) -> dict:
        """Evaluate Prophet on validation set using cross-validation output."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        results = {}
        for state, model in self.models.items():
            state_val = val_df[val_df["state"] == state].sort_values("date")
            if state_val.empty:
                continue
            try:
                prophet_df = self._build_prophet_df(state_val)
                forecast = model.predict(prophet_df.rename(columns={"y": "y_drop"}))
                preds = forecast["yhat"].values
                actuals = state_val["sales"].values
                mae = mean_absolute_error(actuals, preds)
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                results[state] = {"mae": mae, "rmse": rmse}
            except Exception:
                pass
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.models, f"{path}/prophet.pkl")

    def load(self, path: str):
        self.models = joblib.load(f"{path}/prophet.pkl")
        return self
