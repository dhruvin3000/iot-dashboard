# IoT Sensor Data Dashboard

A Streamlit web dashboard that runs a complete IoT data pipeline — from sensor simulation through preprocessing, EDA, and time-series forecasting — all in one interactive app.

Built by combining four standalone Python scripts from the `academics/Python_programs/` folder into a unified pipeline.

## Pipeline Stages

```
Stage 1 — Data Generation    data_generator.py
           │  Simulates DHT11 + BMP180 sensor readings
           │  (Temperature, Humidity, Pressure)
           │  with configurable noise and missing values
           ▼
Stage 2 — Preprocessing      pipeline.py → clean_data()
           │  Forward/backward fill missing values
           │  Clip outliers to physical sensor ranges
           │  Add rolling mean + std features
           ▼
Stage 3 — EDA                pipeline.py → summary_stats(), detect_anomalies()
           │  Summary statistics table
           │  Pearson correlation matrix
           │  Distribution histograms + boxplots
           │  Z-score anomaly detection with visual highlights
           ▼
Stage 4 — Forecasting        forecasting.py → run_all_forecasts()
           │  Moving Average (baseline)
           │  Exponential Smoothing (Holt's linear trend)
           └  ARIMA(2,1,2) (if statsmodels installed)
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the dashboard
```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`

### 3. Use the sidebar to control
- Number of sensor readings (50–500)
- Noise / missing value probability
- Rolling average window
- Forecast horizon
- Anomaly Z-score threshold
- Which sensor to forecast

## File Structure

```
iot-dashboard/
├── dashboard.py          — Streamlit app (main entry point)
├── data_generator.py     — IoT sensor data simulator
├── pipeline.py           — Preprocessing + EDA functions
├── forecasting.py        — Time-series forecasting (MA, Exp. Smoothing, ARIMA)
├── requirements.txt      — Python dependencies
└── README.md
```

## Source Files (from academics/Python_programs/)

| Dashboard module | Original file |
|-----------------|---------------|
| `data_generator.py` | `Data_Collection_from_IoT_Sensors.py` |
| `pipeline.py` (clean) | `Data_Preprocessing_for_IoT_Data.py` |
| `pipeline.py` (EDA) | `Exploratory_Data_Analysis_on_IoT_Dataset.py` |
| `forecasting.py` | `Time_Series_Analysis_of_IoT_Sensor_Data.py` |

## Dashboard Sections

1. **Raw vs Cleaned Data** — side-by-side table comparison
2. **Sensor Time Series** — interactive line charts with rolling average overlay
3. **Summary Statistics** — mean, median, std, min, max per sensor
4. **Correlation Matrix** — Plotly heatmap
5. **Distributions** — histograms + boxplots
6. **Anomaly Detection** — Z-score flagging with red X markers on chart
7. **Forecasting** — historical + multi-method forecast chart + table

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web dashboard framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `plotly` | Interactive charts |
| `scipy` | Z-score for anomaly detection |
| `statsmodels` | Exponential Smoothing + ARIMA |
