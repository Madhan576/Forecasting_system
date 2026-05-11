# 🚀 End-to-End Time Series Forecasting System

A production-ready sales forecasting system that trains multiple ML/DL models, selects the best per state, and serves predictions via a REST API.

---

## 📁 Project Structure

```
forecasting_system/
├── data/
│   └── preprocessor.py          # Data cleaning & feature engineering
├── models/
│   ├── sarima_model.py          # SARIMA / SARIMAX implementation
│   ├── prophet_model.py         # Facebook Prophet
│   ├── xgboost_model.py         # XGBoost with lag features
│   ├── lstm_model.py            # LSTM deep learning
│   └── model_selector.py        # Best model selection & ensemble
├── api/
│   └── server.py                # FastAPI REST API
├── artifacts/                   # Saved model weights & reports (auto-created)
├── results/                     # Forecast CSVs (auto-created)
├── train.py                     # Main training pipeline
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install Prophet
pip install prophet
```

---

## 🏋️ Training

```bash
# Train all models on your dataset
python train.py --data path/to/your_data.xlsx --val_weeks 8 --forecast 8
```

**What it does:**
1. Loads and cleans the dataset
2. Fills missing dates (weekly frequency)
3. Imputes missing sales (forward/backward fill)
4. Engineers 30+ features (lags, rolling stats, calendar, holidays)
5. Trains SARIMA, Prophet, XGBoost, LSTM per state
6. Evaluates all models on held-out 8-week validation set
7. Selects best model per state by RMSE
8. Generates 8-week forecasts and saves CSVs

---

## 🌐 REST API

### Start the server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

Then open: **http://localhost:8000/docs** (Swagger UI)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/states` | List all states |
| `GET`  | `/models` | Model info & scores |
| `POST` | `/forecast` | Forecast a state |
| `POST` | `/forecast/batch` | Batch forecasts |
| `GET`  | `/forecast/{state}/best` | Best model shorthand |
| `GET`  | `/forecast/{state}/compare` | Compare all models |
| `GET`  | `/forecast/{state}/summary` | Summary stats |
| `POST` | `/retrain` | Trigger retraining |

### Example API Call

```bash
# Forecast next 8 weeks for California
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "steps": 8, "model": "best"}'
```

**Response:**
```json
{
  "state": "California",
  "model_used": "XGBoost",
  "steps": 8,
  "generated_at": "2024-01-15T10:30:00",
  "forecast": [
    {"week": 1, "date": "2024-01-22", "predicted_sales": 15420.50, "lower_bound": 13069.42, "upper_bound": 17771.58},
    {"week": 2, "date": "2024-01-29", "predicted_sales": 16234.80, ...},
    ...
  ],
  "model_score": {"mae": 892.3, "rmse": 1124.7}
}
```

---

## 📊 Feature Engineering

| Feature | Description |
|---------|-------------|
| `lag_1w` – `lag_52w` | Sales 1, 2, 4, 8, 12, 26, 52 weeks ago |
| `rolling_mean_4w/8w/13w` | Rolling average (shift-1 to avoid leakage) |
| `rolling_std_4w/8w/13w` | Rolling standard deviation |
| `rolling_max_4w/8w/13w` | Rolling maximum |
| `ewm_4w`, `ewm_12w` | Exponential weighted mean |
| `day_of_week` | Mon=0, Sun=6 |
| `week_of_year` | 1–52 |
| `month`, `quarter` | Calendar period |
| `is_holiday` | US federal holiday flag |
| `is_month_start/end` | Month boundary flags |
| `dow_sin/cos` | Cyclic encoding of day of week |
| `month_sin/cos` | Cyclic encoding of month |

---

## 🤖 Models

### SARIMA
- Automatic order selection via AIC grid search
- Seasonal period = 52 weeks (yearly)
- ADF test for stationarity detection

### Facebook Prophet
- Multiplicative seasonality mode
- US public holidays included
- 95% confidence intervals

### XGBoost
- 300 trees, depth 6, LR 0.05
- Early stopping on validation
- Recursive multi-step forecasting

### LSTM
- 2-layer LSTM: 64 → 32 units
- Dropout 0.2 + BatchNormalization
- 13-week lookback window
- EarlyStopping + ReduceLROnPlateau

---

## ✅ Model Selection Logic

1. All models evaluated on last 8 weeks (held-out, no leakage)
2. Primary metric: **RMSE** (penalizes large errors more)
3. Best model selected per state independently
4. Ensemble weights = softmax of inverse RMSE across all models
5. Fallback: XGBoost (most robust with limited data)

---

## 📈 Time Series Cross-Validation

```
Train data          Validation
|━━━━━━━━━━━━━━━━━━━|━━━━━━━━|
Jan 2019   Nov 2023  Dec 2023 Jan 2024

• No future data leaks into training
• Validation = always the last N weeks chronologically
• Lag features shifted by 1 to prevent look-ahead bias
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Data | pandas, numpy, openpyxl |
| Statistical | statsmodels (SARIMA) |
| ML | xgboost, scikit-learn |
| DL | TensorFlow / Keras (LSTM) |
| Forecasting | Prophet |
| API | FastAPI, Uvicorn |
| Serialization | joblib, JSON, Parquet |

---

## 📧 Author

Madhan — B.E. Computer Science, Vels University, Chennai  
Data Science Assignment — End-to-End Time Series Forecasting System
