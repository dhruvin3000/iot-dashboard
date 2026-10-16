"""
data_generator.py
─────────────────
IoT sensor data simulator.

Generates realistic environmental readings (Temperature, Humidity, Pressure)
with optional noise and missing values — mirroring a real DHT11 + BMP180 setup.

Originally from: Data_Collection_from_IoT_Sensors.py
Extended with:   configurable sample count, seed, and noise injection.
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_sensor_data(
    n_samples: int = 100,
    seed: int = 42,
    noise_prob: float = 0.05,
    start_time: datetime | None = None,
    interval_seconds: int = 2,
) -> pd.DataFrame:
    """
    Generate a DataFrame of simulated IoT sensor readings.

    Parameters
    ----------
    n_samples       : number of readings to generate
    seed            : random seed for reproducibility
    noise_prob      : probability of injecting a missing/bad value (0-1)
    start_time      : timestamp of the first reading (default: now)
    interval_seconds: seconds between readings

    Returns
    -------
    pd.DataFrame with columns:
        Timestamp, Temperature (°C), Humidity (%), Pressure (hPa)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if start_time is None:
        start_time = datetime.now() - timedelta(seconds=n_samples * interval_seconds)

    records = []
    for i in range(n_samples):
        ts = start_time + timedelta(seconds=i * interval_seconds)

        temp = round(rng.uniform(20.0, 35.0), 2)
        hum  = round(rng.uniform(30.0, 90.0), 2)
        pres = round(rng.uniform(950.0, 1050.0), 2)

        # Inject noise / missing values
        if rng.random() < noise_prob:
            temp = None
        if rng.random() < noise_prob:
            hum = rng.choice([None, 200.0, -5.0])  # realistic outlier types

        records.append({
            "Timestamp":          ts,
            "Temperature (°C)":   temp,
            "Humidity (%)":       hum,
            "Pressure (hPa)":     pres,
        })

    return pd.DataFrame(records)


def save_to_csv(df: pd.DataFrame, path: str = "environment_data.csv") -> None:
    """Save sensor DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} records to {path}")


if __name__ == "__main__":
    df = generate_sensor_data(n_samples=200)
    save_to_csv(df)
    print(df.head())
