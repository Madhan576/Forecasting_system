"""
Model Selector & Ensemble Manager
===================================
Compares all models on validation set and selects best per state.
Also supports ensemble averaging for improved robustness.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Optional


class ModelSelector:
    """
    Evaluates SARIMA, Prophet, XGBoost, LSTM on the validation set.
    Selects the best model per state based on RMSE (primary) and MAE (tiebreak).
    Optionally creates a weighted ensemble.
    """

    def __init__(self):
        self.model_scores = {}          # state -> {model: {mae, rmse}}
        self.best_models = {}           # state -> model_name
        self.selection_report = {}      # full report for API
        self.ensemble_weights = {}      # state -> {model: weight}

    def collect_scores(
        self,
        sarima_scores: dict,
        prophet_scores: dict,
        xgboost_scores: dict,
        lstm_scores: dict,
    ):
        """Aggregate evaluation scores from all models."""
        score_maps = {
            "SARIMA": sarima_scores,
            "Prophet": prophet_scores,
            "XGBoost": xgboost_scores,
            "LSTM": lstm_scores,
        }

        all_states = set()
        for scores in score_maps.values():
            all_states.update(scores.keys())

        for state in all_states:
            self.model_scores[state] = {}
            for model_name, scores in score_maps.items():
                if state in scores:
                    self.model_scores[state][model_name] = scores[state]

        print(f"[Selector] Collected scores for {len(self.model_scores)} states.")
        return self

    def select_best(self, metric: str = "rmse"):
        """Pick the best model per state by lowest RMSE (or MAE)."""
        for state, scores in self.model_scores.items():
            if not scores:
                self.best_models[state] = "XGBoost"  # default fallback
                continue

            best_model = min(scores, key=lambda m: scores[m].get(metric, float("inf")))
            self.best_models[state] = best_model

            self.selection_report[state] = {
                "best_model": best_model,
                "scores": {
                    model: {
                        "mae": round(v.get("mae", 0), 2),
                        "rmse": round(v.get("rmse", 0), 2),
                    }
                    for model, v in scores.items()
                },
            }

        print(f"[Selector] Best models selected:")
        from collections import Counter
        counts = Counter(self.best_models.values())
        for model, cnt in counts.most_common():
            print(f"  {model}: {cnt} states")

        return self

    def compute_ensemble_weights(self, metric: str = "rmse"):
        """
        Compute softmax-weighted ensemble based on inverse RMSE.
        Lower error → higher weight.
        """
        for state, scores in self.model_scores.items():
            if not scores:
                continue

            # Inverse metric (lower error = higher weight)
            raw_weights = {}
            for model, v in scores.items():
                err = v.get(metric, float("inf"))
                raw_weights[model] = 1.0 / (err + 1e-8)

            total = sum(raw_weights.values())
            self.ensemble_weights[state] = {
                m: round(w / total, 4) for m, w in raw_weights.items()
            }

        return self

    def get_best_model_name(self, state: str) -> str:
        return self.best_models.get(state, "XGBoost")

    def combine_forecasts(
        self,
        forecasts: Dict[str, pd.DataFrame],   # model_name -> DataFrame
        states: list,
        steps: int = 8,
        strategy: str = "best",               # "best" | "ensemble"
    ) -> pd.DataFrame:
        """
        Combine forecasts from multiple models per state.
        
        strategy='best':     Use only the best model's prediction per state.
        strategy='ensemble': Weighted average across all available models.
        """
        records = []

        for state in states:
            for week in range(1, steps + 1):

                if strategy == "ensemble" and state in self.ensemble_weights:
                    weights = self.ensemble_weights[state]
                    pred = 0.0
                    total_w = 0.0
                    for model_name, weight in weights.items():
                        if model_name not in forecasts:
                            continue
                        model_fc = forecasts[model_name]
                        row = model_fc[
                            (model_fc["state"] == state) & (model_fc["week"] == week)
                        ]
                        if not row.empty:
                            pred += weight * float(row["predicted_sales"].values[0])
                            total_w += weight
                    if total_w > 0:
                        pred /= total_w
                    date = None
                    for model_name in forecasts:
                        row = forecasts[model_name][
                            (forecasts[model_name]["state"] == state) & (forecasts[model_name]["week"] == week)
                        ]
                        if not row.empty:
                            date = row["date"].values[0]
                            break

                else:  # best model
                    best_model = self.get_best_model_name(state)
                    if best_model not in forecasts:
                        best_model = list(forecasts.keys())[0]
                    model_fc = forecasts[best_model]
                    row = model_fc[
                        (model_fc["state"] == state) & (model_fc["week"] == week)
                    ]
                    if row.empty:
                        continue
                    pred = float(row["predicted_sales"].values[0])
                    date = row["date"].values[0]

                records.append({
                    "state": state,
                    "week": week,
                    "date": pd.Timestamp(date) if date is not None else None,
                    "predicted_sales": round(max(0, pred), 2),
                    "strategy": strategy,
                    "model_used": self.get_best_model_name(state) if strategy == "best" else "Ensemble",
                })

        return pd.DataFrame(records)

    def save_report(self, path: str):
        """Save model selection report as JSON."""
        os.makedirs(path, exist_ok=True)
        report = {
            "best_models": self.best_models,
            "scores": self.selection_report,
            "ensemble_weights": self.ensemble_weights,
        }
        with open(f"{path}/model_selection_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Selector] Report saved to {path}/model_selection_report.json")

    def load_report(self, path: str):
        with open(f"{path}/model_selection_report.json") as f:
            report = json.load(f)
        self.best_models = report["best_models"]
        self.selection_report = report["scores"]
        self.ensemble_weights = report.get("ensemble_weights", {})
        return self

    def print_leaderboard(self):
        """Pretty-print model leaderboard."""
        print("\n" + "=" * 60)
        print("MODEL SELECTION LEADERBOARD")
        print("=" * 60)
        for state, info in self.selection_report.items():
            print(f"\n📍 {state}  →  Best: {info['best_model']}")
            for model, metrics in info["scores"].items():
                marker = " ✓" if model == info["best_model"] else "  "
                print(f"  {marker} {model:10s}  MAE={metrics['mae']:8.1f}  RMSE={metrics['rmse']:8.1f}")
        print("=" * 60 + "\n")
