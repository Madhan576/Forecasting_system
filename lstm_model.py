"""
LSTM Deep Learning Model for State-Level Sales Forecasting
===========================================================
Sequence-to-one LSTM with dropout regularization.
Uses sliding window of 13 weeks to predict next week.
Then applies recursive multi-step forecasting.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class LSTMForecaster:
    """LSTM forecaster per state using TensorFlow/Keras."""

    def __init__(self, lookback: int = 13, epochs: int = 80, batch_size: int = 16):
        self.lookback = lookback    # Weeks of history used as input sequence
        self.epochs = epochs
        self.batch_size = batch_size
        self.models = {}            # state -> Keras model
        self.scalers = {}           # state -> MinMaxScaler
        self.model_name = "LSTM"
        self._keras_available = self._check_keras()

    def _check_keras(self) -> bool:
        try:
            import tensorflow as tf
            return True
        except ImportError:
            print("[WARNING] TensorFlow not available. LSTM model disabled.")
            return False

    def _build_model(self, n_features: int = 1):
        """Build LSTM architecture."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            LSTM(64, return_sequences=True,
                 input_shape=(self.lookback, n_features),
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(0.2),
            BatchNormalization(),
            LSTM(32, return_sequences=False,
                 kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss="huber")
        return model

    def _create_sequences(self, data: np.ndarray):
        """Create input-output sequences from time series."""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i: i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train LSTM per state."""
        if not self._keras_available:
            print("[LSTM] Skipped — TensorFlow not available.")
            return self

        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        states = train_df["state"].unique()
        print(f"[LSTM] Fitting {len(states)} state models...")

        for state in states:
            state_df = train_df[train_df["state"] == state].sort_values("date")
            sales = state_df["sales"].values.reshape(-1, 1)

            if len(sales) < self.lookback + 5:
                print(f"  [LSTM] Skipping {state}: insufficient data ({len(sales)} rows).")
                continue

            try:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(sales)

                X, y = self._create_sequences(scaled)
                X = X.reshape(X.shape[0], X.shape[1], 1)

                # Validation split from end of training
                val_size = min(8, len(X) // 5)
                X_train, X_val = X[:-val_size], X[-val_size:]
                y_train, y_val = y[:-val_size], y[-val_size:]

                model = self._build_model(n_features=1)
                callbacks = [
                    EarlyStopping(patience=15, restore_best_weights=True, monitor="val_loss"),
                    ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-5),
                ]
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=callbacks,
                    verbose=0,
                )
                self.models[state] = model
                self.scalers[state] = scaler
                print(f"  [LSTM] {state}: fitted on {len(X_train)} sequences.")
            except Exception as e:
                print(f"  [LSTM] {state}: failed — {e}")

        print(f"[LSTM] Fitted {len(self.models)} / {len(states)} models.")
        return self

    def predict(self, state: str, train_df: pd.DataFrame, steps: int = 8) -> list:
        """Recursive multi-step forecast."""
        if state not in self.models:
            raise ValueError(f"No LSTM model for state: {state}")

        model = self.models[state]
        scaler = self.scalers[state]

        state_df = train_df[train_df["state"] == state].sort_values("date")
        sales = state_df["sales"].values
        last_date = state_df["date"].max()

        # Seed with last `lookback` weeks
        seed = scaler.transform(sales[-self.lookback:].reshape(-1, 1))
        current_sequence = list(seed.flatten())

        predictions_scaled = []
        for _ in range(steps):
            input_seq = np.array(current_sequence[-self.lookback:]).reshape(1, self.lookback, 1)
            pred_scaled = float(model.predict(input_seq, verbose=0)[0][0])
            predictions_scaled.append(pred_scaled)
            current_sequence.append(pred_scaled)

        # Inverse transform
        preds = scaler.inverse_transform(
            np.array(predictions_scaled).reshape(-1, 1)
        ).flatten()
        preds = np.clip(preds, 0, None)  # No negative sales

        future_dates = pd.date_range(
            last_date + pd.Timedelta(weeks=1), periods=steps, freq="W-MON"
        )
        return list(preds), list(future_dates)

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
                        "predicted_sales": round(float(val), 2),
                        "model": self.model_name,
                    })
            except Exception as e:
                print(f"  [LSTM] Predict failed for {state}: {e}")
        return pd.DataFrame(records)

    def evaluate(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
        """Evaluate LSTM on validation set."""
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

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        for state, model in self.models.items():
            safe_state = state.replace(" ", "_")
            model.save(f"{path}/lstm_{safe_state}.keras")
        joblib.dump(self.scalers, f"{path}/lstm_scalers.pkl")

    def load(self, path: str):
        import tensorflow as tf
        import glob
        self.scalers = joblib.load(f"{path}/lstm_scalers.pkl")
        for filepath in glob.glob(f"{path}/lstm_*.keras"):
            state = os.path.basename(filepath).replace("lstm_", "").replace(".keras", "").replace("_", " ")
            self.models[state] = tf.keras.models.load_model(filepath)
        return self
