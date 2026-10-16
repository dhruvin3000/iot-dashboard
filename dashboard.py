"""
dashboard.py — IoT Sensor Data Dashboard
─────────────────────────────────────────
A Streamlit app that walks through the full IoT data pipeline:

  Stage 1 ▸ Data Generation   (data_generator.py)
  Stage 2 ▸ Preprocessing      (pipeline.py — clean, clip, impute)
  Stage 3 ▸ EDA                (pipeline.py — stats, anomalies, correlation)
  Stage 4 ▸ Forecasting        (forecasting.py — MA, Exp. Smoothing, ARIMA)

Run:
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_generator import generate_sensor_data
from forecasting import run_all_forecasts
from pipeline import (
    add_rolling_features,
    clean_data,
    correlation_matrix,
    detect_anomalies,
    summary_stats,
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="IoT Sensor Dashboard",
    page_icon="🌡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar controls ─────────────────────────────────────────────────────────

st.sidebar.title("⚙ Dashboard Controls")

n_samples      = st.sidebar.slider("Number of sensor readings",   50, 500, 200, step=10)
noise_prob     = st.sidebar.slider("Noise / missing value prob.",  0.0, 0.3, 0.05, step=0.01)
rolling_window = st.sidebar.slider("Rolling average window",       2, 20, 5)
forecast_steps = st.sidebar.slider("Forecast horizon (steps)",     5, 30, 10)
z_threshold    = st.sidebar.slider("Anomaly Z-score threshold",    1.5, 4.0, 2.5, step=0.1)
seed           = st.sidebar.number_input("Random seed", value=42, step=1)

sensor_col = st.sidebar.selectbox(
    "Sensor to forecast",
    ["Temperature (°C)", "Humidity (%)", "Pressure (hPa)"],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Pipeline stages:**\n"
    "1. Generate simulated sensor data\n"
    "2. Clean → impute → clip outliers\n"
    "3. Explore → stats, anomalies, correlation\n"
    "4. Forecast → MA, Exp. Smoothing, ARIMA"
)

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🌡 IoT Sensor Data Dashboard")
st.markdown(
    "Full pipeline: **Data Generation → Preprocessing → EDA → Forecasting**  \n"
    "Adjust controls in the sidebar and the dashboard updates instantly."
)

# ─── Stage 1: Generate data ───────────────────────────────────────────────────

with st.spinner("Generating sensor data..."):
    raw_df = generate_sensor_data(
        n_samples=n_samples,
        seed=int(seed),
        noise_prob=noise_prob,
    )

# ─── Stage 2: Preprocessing ───────────────────────────────────────────────────

cleaned_df = clean_data(raw_df)
cleaned_df = add_rolling_features(cleaned_df, window=rolling_window)
ts_df      = cleaned_df.set_index("Timestamp")

st.header("Stage 1 & 2 — Raw vs Cleaned Data")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Data (first 10 rows)")
    st.dataframe(raw_df.head(10), use_container_width=True)
    null_count = raw_df.isnull().sum().sum()
    st.caption(f"Missing / bad values in raw data: **{null_count}**")

with col2:
    st.subheader("Cleaned Data (first 10 rows)")
    st.dataframe(cleaned_df[["Timestamp", "Temperature (°C)", "Humidity (%)", "Pressure (hPa)"]].head(10),
                 use_container_width=True)
    st.caption("All nulls imputed, outliers clipped to sensor ranges.")

# ─── Live sensor charts ───────────────────────────────────────────────────────

st.header("Stage 2 — Sensor Time Series")

sensor_colors = {
    "Temperature (°C)": "red",
    "Humidity (%)":     "royalblue",
    "Pressure (hPa)":   "green",
}

for col, color in sensor_colors.items():
    ma_col = f"{col} MA({rolling_window})"
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ts_df.index, y=ts_df[col],
        mode="lines", name=col,
        line=dict(color=color, width=1.5),
        opacity=0.7,
    ))

    if ma_col in ts_df.columns:
        fig.add_trace(go.Scatter(
            x=ts_df.index, y=ts_df[ma_col],
            mode="lines", name=f"Rolling Avg ({rolling_window})",
            line=dict(color="black", width=2, dash="dash"),
        ))

    fig.update_layout(
        title=col, height=280, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── Stage 3: EDA ─────────────────────────────────────────────────────────────

st.header("Stage 3 — Exploratory Data Analysis")

eda_col1, eda_col2 = st.columns(2)

with eda_col1:
    st.subheader("Summary Statistics")
    stats = summary_stats(cleaned_df)
    st.dataframe(stats, use_container_width=True)

with eda_col2:
    st.subheader("Correlation Matrix")
    corr = correlation_matrix(cleaned_df)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson Correlation",
        height=300,
    )
    fig_corr.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

# Distribution histograms
st.subheader("Distributions")
hist_cols = st.columns(3)
for i, (col, color) in enumerate(sensor_colors.items()):
    with hist_cols[i]:
        fig_hist = px.histogram(
            cleaned_df, x=col, nbins=20,
            color_discrete_sequence=[color],
            title=f"{col} Distribution",
            height=280,
        )
        fig_hist.update_layout(margin=dict(l=5, r=5, t=40, b=5), showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

# Boxplot
st.subheader("Boxplots")
box_data = cleaned_df[["Temperature (°C)", "Humidity (%)", "Pressure (hPa)"]].melt(var_name="Sensor", value_name="Value")
fig_box = px.box(box_data, x="Sensor", y="Value", color="Sensor",
                 color_discrete_map=sensor_colors, height=350)
fig_box.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# Anomaly detection
st.subheader(f"Anomaly Detection (Z-score threshold = {z_threshold})")
anomaly_df = detect_anomalies(cleaned_df, z_threshold=z_threshold)

anomaly_counts = {}
for col in sensor_colors:
    acol = f"{col}_anomaly"
    if acol in anomaly_df.columns:
        count = anomaly_df[acol].sum()
        anomaly_counts[col] = count

m1, m2, m3 = st.columns(3)
for metric_col, (sensor, count) in zip([m1, m2, m3], anomaly_counts.items()):
    with metric_col:
        st.metric(f"Anomalies in {sensor.split(' ')[0]}", count)

for col, color in sensor_colors.items():
    acol = f"{col}_anomaly"
    if acol not in anomaly_df.columns:
        continue

    normal_pts  = anomaly_df[~anomaly_df[acol]]
    anomaly_pts = anomaly_df[anomaly_df[acol]]

    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(
        x=normal_pts["Timestamp"], y=normal_pts[col],
        mode="lines+markers", name="Normal",
        marker=dict(color=color, size=4),
        line=dict(color=color, width=1),
    ))
    if not anomaly_pts.empty:
        fig_a.add_trace(go.Scatter(
            x=anomaly_pts["Timestamp"], y=anomaly_pts[col],
            mode="markers", name="Anomaly",
            marker=dict(color="red", size=10, symbol="x"),
        ))
    fig_a.update_layout(
        title=f"{col} — Anomalies highlighted",
        height=260, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_a, use_container_width=True)

# ─── Stage 4: Forecasting ─────────────────────────────────────────────────────

st.header(f"Stage 4 — Forecasting: {sensor_col}")

with st.spinner("Running forecasts..."):
    try:
        forecasts = run_all_forecasts(ts_df, sensor_col, steps=forecast_steps)

        fig_fc = go.Figure()

        # Historical data
        fig_fc.add_trace(go.Scatter(
            x=ts_df.index, y=ts_df[sensor_col],
            mode="lines", name="Historical",
            line=dict(color=sensor_colors[sensor_col], width=1.5),
            opacity=0.8,
        ))

        # Forecast lines
        fc_colors = ["green", "orange", "purple"]
        for (name, fc_series), fc_color in zip(forecasts.items(), fc_colors):
            fig_fc.add_trace(go.Scatter(
                x=fc_series.index, y=fc_series.values,
                mode="lines+markers", name=name,
                line=dict(color=fc_color, width=2, dash="dash"),
                marker=dict(size=6),
            ))

        # Vertical line at forecast start
        fig_fc.add_vline(
            x=ts_df.index[-1],
            line_dash="dot", line_color="gray",
            annotation_text="Forecast start",
        )

        fig_fc.update_layout(
            title=f"{sensor_col} — Historical + Forecast ({forecast_steps} steps)",
            height=420, margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table
        st.subheader("Forecast Values")
        fc_table = pd.DataFrame(forecasts)
        fc_table.index = fc_table.index.strftime("%Y-%m-%d %H:%M:%S")
        fc_table.index.name = "Timestamp"
        st.dataframe(fc_table, use_container_width=True)

    except Exception as e:
        st.error(f"Forecasting error: {e}")
        st.info("Make sure `statsmodels` is installed: `pip install statsmodels`")

# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "IoT Sensor Dashboard — built from `Data_Collection_from_IoT_Sensors.py`, "
    "`Data_Preprocessing_for_IoT_Data.py`, `Exploratory_Data_Analysis_on_IoT_Dataset.py`, "
    "and `Time_Series_Analysis_of_IoT_Sensor_Data.py`"
)
