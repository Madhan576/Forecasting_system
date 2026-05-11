"""
SARIMA Model for State-Level Sales Forecasting
===============================================
Fits SARIMA(p,d,q)(P,D,Q,s) with automatic order selection via AIC.
"""

import pandas as pd
import numpy as np
import warnings
import joblib
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from itertools import product
warnings.filterwarnings("ignore")


class SARIMAForecaster:
    """SARIMA forecaster trained per state."""

    def __init__(self, seasonal_period: int = 52):
        self.seasonal_period = seasonal_period
        self.models = {}       # state -> fitted model result
        self.orders = {}       # state -> (p,d,q,P,D,Q)
        self.model_name = "SARIMA"

    def _check_stationarity(self, series: pd.Series) -> int:
        """ADF test to determine differencing order d."""
        try:
            p_value = adfuller(series.dropna())[1]
            return 0 if p_value < 0.05 else 1
        except Exception:
            return 1

    def _select_order(self, series: pd.Series):
        """Grid search over small SARIMA space using AIC."""
        d = self._check_stationarity(series)
        best_aic = np.inf
        best_order = (1, d, 1, 0, 0, 0)

        # Reduced grid for speed
        p_vals = [0, 1, 2]
        q_vals = [0, 1, 2]
        P_vals = [0, 1]
        Q_vals = [0, 1]

        for p, q, P, Q in product(p_vals, q_vals, P_vals, Q_vals):
            try:
                model = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(P, 0, Q, self.seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=100)
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, d, q, P, 0, Q)
            except Exception:
                continue

        return best_order

    def fit(self, train_df: pd.DataFrame):
        """Fit a SARIMA model for each state."""
        states = train_df["state"].unique()
        print(f"[SARIMA] Fitting {len(states)} state models...")

        for state in states:
            state_df = (
                train_df[train_df["state"] == state]
                .set_index("date")["sales"]
                .sort_index()
            )

            if len(state_df) < 20:
                print(f"  [SARIMA] Skipping {state}: insufficient data ({len(state_df)} rows)")
                continue

            try:
                order = self._select_order(state_df)
                p, d, q, P, D, Q = order
                model = SARIMAX(
                    state_df,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, self.seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=200)
                self.models[state] = result
                self.orders[state] = order
                print(f"  [SARIMA] {state}: order=({p},{d},{q})({P},{D},{Q},{self.seasonal_period}), AIC={result.aic:.1f}")
            except Exception as e:
                print(f"  [SARIMA] {state}: failed — {e}")

        print(f"[SARIMA] Fitted {len(self.models)} / {len(states)} models.")
        return self

    def predict(self, state: str, steps: int = 8) -> pd.Series:
        """Forecast next N weeks for a given state."""
        if state not in self.models:
            raise ValueError(f"No SARIMA model for state: {state}")
        result = self.models[state]
        forecast = result.forecast(steps=steps)
        forecast = forecast.clip(lower=0)  # No negative sales
        return forecast

    def predict_all(self, states: list = None, steps: int = 8) -> pd.DataFrame:
        """Forecast for all (or specified) states."""
        states = states or list(self.models.keys())
        records = []
        for state in states:
            if state not in self.models:
                continue
            fc = self.predict(state, steps)
            for i, (date, val) in enumerate(fc.items()):
                records.append({
                    "state": state,
                    "week": i + 1,
                    "date": date,
                    "predicted_sales": round(float(val), 2),
                    "model": self.model_name,
                })
        return pd.DataFrame(records)

    def evaluate(self, val_df: pd.DataFrame) -> dict:
        """Evaluate SARIMA on validation set."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        results = {}
        for state, model in self.models.items():
            state_val = val_df[val_df["state"] == state].sort_values("date")
            if state_val.empty:
                continue
            try:
                preds = model.predict(
                    start=state_val["date"].iloc[0],
                    end=state_val["date"].iloc[-1]
                ).values
                actuals = state_val["sales"].values[:len(preds)]
                mae = mean_absolute_error(actuals, preds)
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                results[state] = {"mae": mae, "rmse": rmse}
            except Exception:
                pass
        return results

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump({"models": self.models, "orders": self.orders}, f"{path}/sarima.pkl")

    def load(self, path: str):
        data = joblib.load(f"{path}/sarima.pkl")
        self.models = data["models"]
        self.orders = data["orders"]
        return self
