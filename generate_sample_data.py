"""
Synthetic Data Generator
========================
Generates realistic weekly sales data for US states,
mimicking the structure of the assignment dataset.
Includes trends, seasonality, noise, and occasional missing values.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

STATES = [
    "California", "Texas", "New York", "Florida", "Illinois",
    "Pennsylvania", "Ohio", "Georgia", "North Carolina", "Michigan"
]

def generate_sales_data(output_path: str = "data/sales_data.xlsx"):
    """Generate weekly sales data for 10 states over 3 years."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dates = pd.date_range("2021-01-04", periods=156, freq="W-MON")  # 3 years of weekly data
    records = []

    for state in STATES:
        # Base sales level varies by state
        base = np.random.uniform(8000, 25000)
        trend = np.random.uniform(10, 80)   # Weekly upward trend

        for i, date in enumerate(dates):
            # Trend component
            trend_val = base + trend * i

            # Yearly seasonality (peak in Nov-Dec, dip in Jan-Feb)
            yearly = 0.15 * trend_val * np.sin(2 * np.pi * (i - 10) / 52)

            # Weekly noise
            noise = np.random.normal(0, trend_val * 0.05)

            # Holiday boost (week 49-51 = Thanksgiving & Christmas shopping)
            week_of_year = date.isocalendar().week
            holiday_boost = trend_val * 0.3 if week_of_year in [49, 50, 51, 52] else 0

            sales = max(0, trend_val + yearly + noise + holiday_boost)

            # Randomly introduce missing values (~3%)
            if np.random.random() < 0.03:
                sales = np.nan

            records.append({
                "Date": date,
                "State": state,
                "Sales": round(sales, 2),
            })

    df = pd.DataFrame(records)
    df.to_excel(output_path, index=False)
    print(f"[DataGen] Generated {len(df)} records for {len(STATES)} states → {output_path}")
    return df


if __name__ == "__main__":
    generate_sales_data("data/sales_data.xlsx")
    print("Synthetic dataset ready.")
