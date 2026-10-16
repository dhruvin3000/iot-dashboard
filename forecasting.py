"""
forecasting.py
──────────────
Time-series forecasting for IoT sensor data.

Provides two forecasting methods:
  - Moving Average (simple baseline)
  - Exponential Smoothing (Holt's linear trend)

ARIMA is included as an optional method — requires statsmodels.

Originally from: Time_Series_Analysis_of_IoT_Sensor_Data.py
"""

import numpy as np
import pandas as pd


# ─── Moving Average Forecast ──────────────────────────────────────────────────

def moving_average_forecast(series: pd.Series, window: int = 5, steps: int = 10) -> pd.Series:
    """
    Naive moving-average forecast.
    Last <window> values averaged, projected flat for <steps> future steps.

    Parameters
    ----------
    series : historical time series (pd.Series with DatetimeIndex)
    window : look-back window size
    steps  : number of future steps to forecast

    Returns
    -------
    pd.Series of forecasted values with a future DatetimeIndex
    """
    if len(series) < window:
        raise ValueError(f"Series too short for window={window}. Need at least {window} points.")

    last_mean = series.iloc[-window:].mean()
    freq = _infer_freq(series)
    future_index = pd.date_range(start=series.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.Series([round(last_mean, 2)] * steps, index=future_index, name="MA Forecast")


# ─── Exponential Smoothing Forecast ──────────────────────────────────────────

def exponential_smoothing_forecast(series: pd.Series, steps: int = 10) -> pd.Series:
    """
    Holt's double exponential smoothing (additive trend, no seasonality).
    Falls back to simple exponential smoothing if statsmodels is unavailable.

    Parameters
    ----------
    series : historical time series
    steps  : number of future steps

    Returns
    -------
    pd.Series of forecasted values
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(series, trend="add", seasonal=None)
        fit   = model.fit(optimized=True)
        forecast = fit.forecast(steps)
        forecast.name = "Exp. Smoothing Forecast"
        return forecast.round(2)
    except ImportError:
        # Graceful fallback: simple weighted average
        return moving_average_forecast(series, window=min(10, len(series)), steps=steps)


# ─── ARIMA Forecast (optional) ───────────────────────────────────────────────

def arima_forecast(series: pd.Series, order: tuple = (2, 1, 2), steps: int = 10) -> pd.Series | None:
    """
    ARIMA forecast. Returns None if statsmodels is not installed.

    Parameters
    ----------
    series : historical time series
    order  : (p, d, q) ARIMA order
    steps  : forecast horizon

    Returns
    -------
    pd.Series of forecasted values, or None
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model    = ARIMA(series, order=order)
        fit      = model.fit()
        forecast = fit.forecast(steps)
        forecast.name = f"ARIMA{order} Forecast"
        return forecast.round(2)
    except ImportError:
        return None
    except Exception:
        # ARIMA can fail on short/noisy series — degrade gracefully
        return None


# ─── Helper ──────────────────────────────────────────────────────────────────

def _infer_freq(series: pd.Series) -> str:
    """Infer frequency string from a DatetimeIndex, default '2s'."""
    try:
        inferred = pd.infer_freq(series.index)
        return inferred or "2s"
    except Exception:
        return "2s"


def run_all_forecasts(df: pd.DataFrame, column: str, steps: int = 10) -> dict:
    """
    Run all available forecast methods on a single sensor column.

    Parameters
    ----------
    df     : cleaned DataFrame with DatetimeIndex
    column : column name to forecast
    steps  : forecast horizon

    Returns
    -------
    dict mapping method name → pd.Series of forecasts
    """
    series = df[column].dropna()
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for forecasting.")

    results = {}
    results["Moving Average"]         = moving_average_forecast(series, steps=steps)
    results["Exponential Smoothing"]   = exponential_smoothing_forecast(series, steps=steps)

    arima = arima_forecast(series, steps=steps)
    if arima is not None:
        results["ARIMA(2,1,2)"] = arima

    return results


if __name__ == "__main__":
    from data_generator import generate_sensor_data
    from pipeline import clean_data

    raw     = generate_sensor_data(n_samples=100)
    cleaned = clean_data(raw).set_index("Timestamp")

    print("Forecasting Temperature (next 10 steps):")
    forecasts = run_all_forecasts(cleaned, "Temperature (°C)", steps=10)
    for name, fc in forecasts.items():
        print(f"\n  {name}:")
        print(fc.to_string())
