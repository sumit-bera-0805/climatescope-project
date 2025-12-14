# app/app.py

import sys
import os
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- ADD PROJECT ROOT TO PYTHON PATH ---
# This ensures `src` can be imported when running from Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import (
    load_parquet_cached,
    ensure_iso3,
    aggregate_time,
    detect_extremes,
    compute_kpis,
    cluster_stations,
    simple_forecast_sarimax_or_rolling,
)

# --- CONFIG ---
st.set_page_config(page_title="ClimateScope", layout="wide")
ROOT = Path(__file__).parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily.parquet"
MONTHLY_PATH = ROOT / "data" / "processed" / "monthly.parquet"

st.title("🌍 ClimateScope")

# --- Load datasets ---
df_daily = load_parquet_cached(DAILY_PATH)
df_monthly = load_parquet_cached(MONTHLY_PATH)

# Ensure `date` column exists
for df in (df_daily, df_monthly):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "month" in df.columns:
        df["date"] = pd.to_datetime(df["month"], errors="coerce")

# Add iso3 codes
country_iso_map = ensure_iso3(pd.concat([df_daily, df_monthly], ignore_index=True))

# --- Sidebar controls ---
st.sidebar.header("Controls")
view = st.sidebar.radio(
    "Section",
    ["Overview", "Maps", "Time Series", "Seasonal", "Extremes", "Compare", "Station Clustering", "Forecasting", "Settings"]
)

VARIABLES = ["temperature", "precipitation", "wind_speed", "humidity"]
var = st.sidebar.selectbox("Variable", VARIABLES)

min_date = min(df_monthly["date"].min(), df_daily["date"].min())
max_date = max(df_monthly["date"].max(), df_daily["date"].max())
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date))

all_countries = sorted(pd.concat([df_daily["country"], df_monthly["country"]]).dropna().unique())
countries = st.sidebar.multiselect("Countries (choose 0 for all)", options=all_countries, default=all_countries[:3])

agg = st.sidebar.radio("Aggregation", ["Daily", "Monthly", "Yearly"])
smooth = st.sidebar.slider("Smoothing window (periods)", 0, 24, 1)

# --- Helper to filter data ---
def apply_filters(df):
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    if countries:
        mask &= df["country"].isin(countries)
    return df.loc[mask].copy()

# === Overview page ===
if view == "Overview":
    st.header("Overview & KPIs")
    df_f = apply_filters(df_monthly if agg != "Daily" else df_daily)
    kpi_df = compute_kpis(df_f, var)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean", f"{kpi_df['mean']:.2f}")
    c2.metric("Median", f"{kpi_df['median']:.2f}")
    c3.metric("Std Dev", f"{kpi_df['std']:.2f}")
    c4.metric("Max", f"{kpi_df['max']:.2f}")

    st.markdown("**Distribution**")
    fig = px.histogram(df_f, x=var, nbins=60, marginal="box", title=f"{var} distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Global monthly climatology (sample)**")
    clim = df_monthly.groupby(df_monthly["date"].dt.month)[var].mean().reset_index()
    clim["month_name"] = clim["date"].apply(lambda x: x)  # month number
    fig2 = px.line(clim, x="date", y=var, title="Climatology (month number vs avg)")
    st.plotly_chart(fig2, use_container_width=True)

# === Maps page ===
if view == "Maps":
    st.header("Maps: Choropleth & Station Map")
    st.subheader("Choropleth — country average")
    country_df = df_monthly.groupby("country")[var].mean().reset_index().rename(columns={var: "mean_var"})
    country_df["iso_a3"] = country_df["country"].map(country_iso_map).fillna("")
    choro = px.choropleth(
        country_df.dropna(subset=["iso_a3"]),
        locations="iso_a3",
        color="mean_var",
        hover_name="country",
        color_continuous_scale="thermal",
        title=f"Mean {var} by country"
    )
    st.plotly_chart(choro, use_container_width=True)

    st.subheader("Station map (points sized/colored by variable)")
    df_points = apply_filters(df_daily)
    if len(df_points) > 5000:
        df_points = df_points.sample(5000, random_state=42)
        st.caption("Sampling 5k points for performance.")
    map_fig = px.scatter_mapbox(df_points, lat="lat", lon="lon", color=var, size=var,
                                hover_data=["station_id","country","date"], zoom=1)
    map_fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(map_fig, use_container_width=True)

# === Time Series page ===
if view == "Time Series":
    st.header("Time Series — Multi-series & smoothing")
    df_ts = apply_filters(df_monthly if agg != "Daily" else df_daily)
    countries_for_plot = st.multiselect(
        "Overlay Countries (empty = use selected countries)",
        options=all_countries,
        default=countries or all_countries[:2]
    )
    if not countries_for_plot:
        countries_for_plot = countries or all_countries[:2]
    df_plot = df_ts[df_ts["country"].isin(countries_for_plot)].groupby(["date","country"])[var].mean().reset_index()
    if smooth > 0:
        df_plot = df_plot.sort_values(["country","date"]).groupby("country").apply(
            lambda g: g.assign(**{var: g[var].rolling(smooth, min_periods=1).mean()})
        ).reset_index(drop=True)
    fig = px.line(df_plot, x="date", y=var, color="country", title=f"{var} over time")
    st.plotly_chart(fig, use_container_width=True)

# === Other pages (Seasonal, Extremes, Compare, Station Clustering, Forecasting) ===
# ... keep the rest of your code unchanged ...

# Footer
st.markdown("---")
st.caption("dashboard — built with Streamlit, Plotly, scikit-learn, and statsmodels.")
