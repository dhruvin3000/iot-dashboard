"""
pipeline.py
───────────
Data preprocessing and EDA functions for IoT sensor data.

Stage 1 — Cleaning:   handle missing values, clip outliers
Stage 2 — EDA:        summary stats, correlation, anomaly detection

Originally from:
  Data_Preprocessing_for_IoT_Data.py
  Exploratory_Data_Analysis_on_IoT_Dataset.py
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─── Stage 1: Preprocessing ──────────────────────────────────────────────────

SENSOR_RANGES = {
    "Temperature (°C)": (15.0, 45.0),
    "Humidity (%)":     (10.0, 100.0),
    "Pressure (hPa)":   (900.0, 1100.0),
}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw IoT sensor DataFrame:
      1. Forward-fill then backward-fill missing values
      2. Clip outliers to realistic physical ranges
      3. Round to 2 decimal places

    Parameters
    ----------
    df : raw DataFrame from data_generator (may have None/NaN values)

    Returns
    -------
    Cleaned DataFrame (no NaNs, values within sensor ranges)
    """
    cleaned = df.copy()

    for col, (lo, hi) in SENSOR_RANGES.items():
        if col not in cleaned.columns:
            continue
        # Cast to float (None → NaN)
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        # Replace out-of-range values with NaN before filling
        cleaned[col] = cleaned[col].where(
            cleaned[col].between(lo - 20, hi + 20), other=np.nan
        )
        # Impute: forward-fill, then backward-fill any remaining NaN
        cleaned[col] = cleaned[col].ffill().bfill()
        # Hard clip to valid sensor range
        cleaned[col] = cleaned[col].clip(lo, hi).round(2)

    # Ensure Timestamp is datetime
    if "Timestamp" in cleaned.columns:
        cleaned["Timestamp"] = pd.to_datetime(cleaned["Timestamp"])

    return cleaned


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Add rolling mean and rolling std columns for each sensor metric.
    Useful for smoothing and anomaly detection.
    """
    result = df.copy()
    for col in SENSOR_RANGES:
        if col in result.columns:
            result[f"{col} MA({window})"]  = result[col].rolling(window, min_periods=1).mean().round(2)
            result[f"{col} Std({window})"] = result[col].rolling(window, min_periods=1).std().round(2)
    return result


# ─── Stage 2: EDA ────────────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, median, std, min, max for each numeric sensor column.

    Returns a DataFrame (metrics as columns, sensors as rows).
    """
    numeric_cols = [c for c in SENSOR_RANGES if c in df.columns]
    records = []
    for col in numeric_cols:
        series = df[col].dropna()
        records.append({
            "Sensor":  col,
            "Mean":    round(series.mean(), 2),
            "Median":  round(series.median(), 2),
            "Std Dev": round(series.std(), 2),
            "Min":     round(series.min(), 2),
            "Max":     round(series.max(), 2),
            "Count":   len(series),
        })
    return pd.DataFrame(records).set_index("Sensor")


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag anomalies using Z-score method.
    Any reading more than z_threshold standard deviations from the mean
    is marked as an anomaly.

    Returns DataFrame with added boolean columns: <col>_anomaly
    """
    result = df.copy()
    for col in SENSOR_RANGES:
        if col not in result.columns:
            continue
        z_scores = np.abs(scipy_stats.zscore(result[col].dropna()))
        anomaly_mask = pd.Series(False, index=result.index)
        anomaly_mask.loc[result[col].dropna().index] = z_scores > z_threshold
        result[f"{col}_anomaly"] = anomaly_mask
    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlation matrix for the sensor columns."""
    numeric_cols = [c for c in SENSOR_RANGES if c in df.columns]
    return df[numeric_cols].corr().round(3)


if __name__ == "__main__":
    from data_generator import generate_sensor_data

    raw = generate_sensor_data(n_samples=100)
    print("Raw data (first 5 rows):")
    print(raw.head())

    cleaned = clean_data(raw)
    print("\nCleaned data (first 5 rows):")
    print(cleaned.head())

    print("\nSummary statistics:")
    print(summary_stats(cleaned))

    print("\nCorrelation matrix:")
    print(correlation_matrix(cleaned))
