"""
XGBoost Model for State-Level Sales Forecasting
================================================
Uses lag features, rolling statistics, and calendar features.
Employs recursive multi-step forecasting for 8-week horizon.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


class XGBoostForecaster:
    """XGBoost forecaster with lag-based recursive multi-step prediction."""

    def __init__(self):
        self.models = {}          # state -> trained model
        self.feature_cols = []
        self.label_encoders = {}
        self.model_name = "XGBoost"
        self.best_params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }

    def _prepare_features(self, df: pd.DataFrame, state: str) -> pd.DataFrame:
        """Select and prepare features for a given state's data."""
        state_df = df[df["state"] == state].sort_values("date").copy()

        # Drop rows with NaN lag features (early rows)
        lag_cols = [c for c in state_df.columns if c.startswith("lag_")]
        state_df = state_df.dropna(subset=lag_cols[:3])  # require at least lag_1w, lag_2w, lag_4w

        # Encode day of week, month as cyclic features
        state_df["dow_sin"] = np.sin(2 * np.pi * state_df["day_of_week"] / 7)
        state_df["dow_cos"] = np.cos(2 * np.pi * state_df["day_of_week"] / 7)
        state_df["month_sin"] = np.sin(2 * np.pi * state_df["month"] / 12)
        state_df["month_cos"] = np.cos(2 * np.pi * state_df["month"] / 12)
        state_df["week_sin"] = np.sin(2 * np.pi * state_df["week_of_year"] / 52)
        state_df["week_cos"] = np.cos(2 * np.pi * state_df["week_of_year"] / 52)

        return state_df

    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get all numeric feature columns (exclude target and metadata)."""
        exclude = {"date", "state", "sales"}
        return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train XGBoost model per state with early stopping."""
        states = train_df["state"].unique()
        print(f"[XGBoost] Fitting {len(states)} state models...")

        for state in states:
            state_train = self._prepare_features(train_df, state)

            if len(state_train) < 15:
                print(f"  [XGBoost] Skipping {state}: insufficient data.")
                continue

            feature_cols = self._get_feature_columns(state_train)
            self.feature_cols = feature_cols

            X_train = state_train[feature_cols].fillna(0)
            y_train = state_train["sales"]

            eval_set = [(X_train, y_train)]
            if val_df is not None:
                state_val = self._prepare_features(val_df, state)
                if not state_val.empty:
                    X_val = state_val[feature_cols].fillna(0)
                    y_val = state_val["sales"]
                    eval_set.append((X_val, y_val))

            model = xgb.XGBRegressor(**self.best_params, early_stopping_rounds=30, eval_metric="rmse")
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False,
            )
            self.models[state] = model
            print(f"  [XGBoost] {state}: fitted on {len(X_train)} samples. Best iter: {model.best_iteration}")

        print(f"[XGBoost] Fitted {len(self.models)} / {len(states)} models.")
        return self

    def predict(self, state: str, train_df: pd.DataFrame, steps: int = 8) -> list:
        """
        Recursive multi-step forecast for next N weeks.
        Uses previous predictions as lag inputs for subsequent steps.
        """
        if state not in self.models:
            raise ValueError(f"No XGBoost model for state: {state}")

        model = self.models[state]
        state_history = self._prepare_features(train_df, state).sort_values("date")
        last_date = state_history["date"].max()
        sales_series = list(state_history["sales"].values)

        predictions = []
        future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=steps, freq="W-MON")

        for i, future_date in enumerate(future_dates):
            # Build feature row
            row = {
                "day_of_week": future_date.dayofweek,
                "week_of_year": future_date.isocalendar().week,
                "month": future_date.month,
                "quarter": future_date.quarter,
                "year": future_date.year,
                "is_month_start": int(future_date.is_month_start),
                "is_month_end": int(future_date.is_month_end),
                "is_quarter_end": int(future_date.is_quarter_end),
                "is_holiday": 0,  # Simplified for prediction
                "dow_sin": np.sin(2 * np.pi * future_date.dayofweek / 7),
                "dow_cos": np.cos(2 * np.pi * future_date.dayofweek / 7),
                "month_sin": np.sin(2 * np.pi * future_date.month / 12),
                "month_cos": np.cos(2 * np.pi * future_date.month / 12),
                "week_sin": np.sin(2 * np.pi * int(future_date.isocalendar().week) / 52),
                "week_cos": np.cos(2 * np.pi * int(future_date.isocalendar().week) / 52),
            }

            # Lag features using extended sales series
            all_sales = sales_series + predictions
            lag_map = {"lag_1w": 1, "lag_2w": 2, "lag_4w": 4, "lag_8w": 8,
                       "lag_12w": 12, "lag_26w": 26, "lag_52w": 52}
            for col, lag in lag_map.items():
                row[col] = all_sales[-lag] if len(all_sales) >= lag else 0

            # Rolling features
            for window, col_prefix in [(4, "rolling_mean_4w"), (8, "rolling_mean_8w"), (13, "rolling_mean_13w")]:
                subset = all_sales[-window:] if len(all_sales) >= window else all_sales
                row[col_prefix] = np.mean(subset)
                row[col_prefix.replace("mean", "std")] = np.std(subset) if len(subset) > 1 else 0
                row[col_prefix.replace("mean", "max")] = np.max(subset) if subset else 0

            row["ewm_4w"] = pd.Series(all_sales).ewm(span=4).mean().iloc[-1]
            row["ewm_12w"] = pd.Series(all_sales).ewm(span=12).mean().iloc[-1]
            row["yoy_change"] = 0

            feature_row = pd.DataFrame([row])
            # Align with training feature columns
            for col in self.feature_cols:
                if col not in feature_row:
                    feature_row[col] = 0
            feature_row = feature_row[self.feature_cols].fillna(0)

            pred = float(model.predict(feature_row)[0])
            pred = max(0, pred)  # No negative sales
            predictions.append(pred)

        return predictions, list(future_dates)

    def predict_all(self, train_df: pd.DataFrame, states: list = None, steps: int = 8) -> pd.DataFrame:
        """Forecast for all (or specified) states."""
        states = states or list(self.models.keys())
        records = []
        for state in states:
            if state not in self.models:
                continue
            try:
                preds, dates = self.predict(state, train_df, steps)
                for i, (date, val) in enumerate(zip(dates, preds)):
                    records.append({
                        "state": state,
                        "week": i + 1,
                        "date": date,
                        "predicted_sales": round(val, 2),
                        "model": self.model_name,
                    })
            except Exception as e:
                print(f"  [XGBoost] Predict failed for {state}: {e}")
        return pd.DataFrame(records)

    def evaluate(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        """Evaluate XGBoost on validation data."""
        results = {}
        for state in self.models.keys():
            state_val = val_df[val_df["state"] == state].sort_values("date")
            if state_val.empty:
                continue
            try:
                preds, _ = self.predict(state, train_df, steps=len(state_val))
                actuals = state_val["sales"].values[:len(preds)]
                mae = mean_absolute_error(actuals, preds)
                rmse = np.sqrt(mean_squared_error(actuals, preds))
                results[state] = {"mae": mae, "rmse": rmse}
            except Exception:
                pass
        return results

    def get_feature_importance(self, state: str) -> pd.DataFrame:
        """Return feature importance for a state's model."""
        if state not in self.models:
            return pd.DataFrame()
        model = self.models[state]
        importance = model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importance
        }).sort_values("importance", ascending=False)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        joblib.dump({"models": self.models, "feature_cols": self.feature_cols}, f"{path}/xgboost.pkl")

    def load(self, path: str):
        data = joblib.load(f"{path}/xgboost.pkl")
        self.models = data["models"]
        self.feature_cols = data["feature_cols"]
        return self
