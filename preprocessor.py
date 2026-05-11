"""
Data Preprocessing & Feature Engineering Module
================================================
Handles missing dates, values, and creates all required features.
"""

import pandas as pd
import numpy as np
import holidays
import warnings
warnings.filterwarnings("ignore")


class DataPreprocessor:
    """Handles all data cleaning and feature engineering."""

    def __init__(self, country: str = "US"):
        self.country = country
        self.us_holidays = holidays.US()

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load dataset from Excel or CSV."""
        if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        print(f"[INFO] Loaded {len(df)} rows, {df.shape[1]} columns.")
        return df

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to expected format."""
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Try to detect date and sales columns
        date_cols = [
            c for c in df.columns if "date" in c or "week" in c or "time" in c]
        sales_cols = [
            c for c in df.columns if "sale" in c or "revenue" in c or "qty" in c or "quantity" in c]
        state_cols = [
            c for c in df.columns if "state" in c or "region" in c or "location" in c]

        rename_map = {}
        if date_cols:
            rename_map[date_cols[0]] = "date"
        if sales_cols:
            rename_map[sales_cols[0]] = "sales"
        if state_cols:
            rename_map[state_cols[0]] = "state"

        df = df.rename(columns=rename_map)
        print(f"[INFO] Columns after standardization: {list(df.columns)}")
        return df

    def handle_missing_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in date series per state with zero sales."""
        df["date"] = pd.to_datetime(df["date"])
        all_states = df["state"].unique()
        date_range = pd.date_range(
            df["date"].min(), df["date"].max(), freq="W-MON")

        full_dfs = []
        for state in all_states:
            state_df = df[df["state"] == state].set_index("date")
            state_df = state_df.reindex(date_range)
            state_df["state"] = state
            state_df.index.name = "date"
            full_dfs.append(state_df.reset_index())

        result = pd.concat(full_dfs, ignore_index=True)
        print(f"[INFO] After date filling: {len(result)} rows (was {len(df)})")
        return result

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing sales using forward-fill then backward-fill."""
        missing_before = df["sales"].isna().sum()
        df = df.sort_values(["state", "date"])
        df["sales"] = df.groupby("state")["sales"].transform(
            lambda x: x.ffill().bfill().fillna(0)
        )
        missing_after = df["sales"].isna().sum()
        print(f"[INFO] Missing values: {missing_before} → {missing_after}")
        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all time-series features (lags, rolling, calendar, holidays)."""
        df = df.sort_values(["state", "date"]).reset_index(drop=True)

        # Calendar features
        df["day_of_week"] = df["date"].dt.dayofweek          # 0=Monday
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter
        df["year"] = df["date"].dt.year
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
        df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

        # Holiday flag
        df["is_holiday"] = df["date"].apply(
            lambda d: 1 if d in self.us_holidays else 0
        )

        # Lag features per state
        for lag in [1, 2, 4, 8, 12, 26, 52]:  # weeks
            df[f"lag_{lag}w"] = df.groupby("state")["sales"].shift(lag)

        # Rolling statistics (4-week, 8-week, 13-week windows)
        for window in [4, 8, 13]:
            df[f"rolling_mean_{window}w"] = (
                df.groupby("state")["sales"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )
            df[f"rolling_std_{window}w"] = (
                df.groupby("state")["sales"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
            )
            df[f"rolling_max_{window}w"] = (
                df.groupby("state")["sales"]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
            )

        # Exponential weighted mean
        df["ewm_4w"] = df.groupby("state")["sales"].transform(
            lambda x: x.shift(1).ewm(span=4, min_periods=1).mean()
        )
        df["ewm_12w"] = df.groupby("state")["sales"].transform(
            lambda x: x.shift(1).ewm(span=12, min_periods=1).mean()
        )

        # Year-over-year sales (52-week lag already covers this)
        df["yoy_change"] = df.groupby(
            "state")["sales"].pct_change(periods=52).fillna(0)

        print(
            f"[INFO] Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def train_val_split(self, df: pd.DataFrame, val_weeks: int = 8):
        """
        Time-series aware split — last N weeks as validation.
        No data leakage: validation always comes after training.
        """
        cutoff = df["date"].max() - pd.Timedelta(weeks=val_weeks)
        train = df[df["date"] <= cutoff].copy()
        val = df[df["date"] > cutoff].copy()
        print(
            f"[INFO] Train: {len(train)} rows | Val: {len(val)} rows | Cutoff: {cutoff.date()}")
        return train, val

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Return list of usable feature columns (excludes date, state, sales)."""
        exclude = {"date", "state", "sales"}
        return [c for c in df.columns if c not in exclude]

    def preprocess_pipeline(self, filepath: str, val_weeks: int = 8):
        """Full preprocessing pipeline."""
        df = self.load_data(filepath)
        df = self.standardize_columns(df)
        df = self.handle_missing_dates(df)
        df = self.handle_missing_values(df)
        df = self.create_features(df)
        train, val = self.train_val_split(df, val_weeks=val_weeks)
        return df, train, val
