# app/app.py
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
st.set_page_config(page_title="ClimateScope ", layout="wide")
ROOT = Path(__file__).parent.parent
DAILY_PATH = ROOT / "data" / "processed" / "daily.parquet"
MONTHLY_PATH = ROOT / "data" / "processed" / "monthly.parquet"

st.title("🌍 ClimateScope")

# Load datasets
df_daily = load_parquet_cached(DAILY_PATH)
df_monthly = load_parquet_cached(MONTHLY_PATH)

# Ensure `date` is datetime and unify columns
for df in (df_daily, df_monthly):
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "month" in df.columns:
        df["date"] = pd.to_datetime(df["month"], errors="coerce")

# Add iso3 using pycountry if available (for choropleth)
country_iso_map = ensure_iso3(pd.concat([df_daily, df_monthly], ignore_index=True))

# --- Sidebar controls (common) ---
st.sidebar.header("Controls")
view = st.sidebar.radio("Section", ["Overview", "Maps", "Time Series", "Seasonal", "Extremes", "Compare", "Station Clustering", "Forecasting", "Settings"])

# variable selection
VARIABLES = ["temperature", "precipitation", "wind_speed", "humidity"]
var = st.sidebar.selectbox("Variable", VARIABLES)

# date range (use monthly for range if available otherwise daily)
min_date = min(df_monthly["date"].min(), df_daily["date"].min())
max_date = max(df_monthly["date"].max(), df_daily["date"].max())
start_date, end_date = st.sidebar.date_input("Date range", value=(min_date, max_date))

# region / country multiselect
all_countries = sorted(pd.concat([df_daily["country"], df_monthly["country"]]).dropna().unique())
countries = st.sidebar.multiselect("Countries (choose 0 for all)", options=all_countries, default=all_countries[:3])

# aggregation level
agg = st.sidebar.radio("Aggregation", ["Daily", "Monthly", "Yearly"])
smooth = st.sidebar.slider("Smoothing window (periods)", 0, 24, 1)

# helper: filter
def apply_filters(df):
    m = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    if countries:
        m &= df["country"].isin(countries)
    return df.loc[m].copy()

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
    clim["month_name"] = clim["date"].apply(lambda x: x)  # already month number in index
    fig2 = px.line(clim, x="date", y=var, title="Climatology (month number vs avg)")
    st.plotly_chart(fig2, use_container_width=True)

# === Maps page ===
if view == "Maps":
    st.header("Maps: Choropleth & Station Map")
    st.subheader("Choropleth — country average")
    country_df = df_monthly.groupby("country")[var].mean().reset_index().rename(columns={var: "mean_var"})
    country_df["iso_a3"] = country_df["country"].map(country_iso_map).fillna("")

    choro = px.choropleth(country_df.dropna(subset=["iso_a3"]), locations="iso_a3", color="mean_var",
                          hover_name="country", color_continuous_scale="thermal",
                          title=f"Mean {var} by country")
    st.plotly_chart(choro, use_container_width=True)

    st.subheader("Station map (points sized/colored by variable)")
    df_points = apply_filters(df_daily)
    # sample if too many
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
    # allow multi-country overlay
    countries_for_plot = st.multiselect("Overlay Countries (empty = use selected countries)", options=all_countries, default=countries or all_countries[:2])
    if not countries_for_plot:
        countries_for_plot = countries or all_countries[:2]
    df_plot = df_ts[df_ts["country"].isin(countries_for_plot)].groupby(["date","country"])[var].mean().reset_index()
    # smoothing per group
    if smooth > 0:
        df_plot = df_plot.sort_values(["country","date"]).groupby("country").apply(lambda g: g.assign(**{var: g[var].rolling(smooth, min_periods=1).mean()})).reset_index(drop=True)
    fig = px.line(df_plot, x="date", y=var, color="country", title=f"{var} over time")
    st.plotly_chart(fig, use_container_width=True)

# === Seasonal page ===
if view == "Seasonal":
    st.header("Seasonal Heatmap (month vs year)")
    df_s = apply_filters(df_monthly)
    df_s["year"] = df_s["date"].dt.year
    df_s["month"] = df_s["date"].dt.month
    sel_country = st.selectbox("Heatmap country", options=(countries or all_countries))
    hm = df_s[df_s["country"] == sel_country].pivot_table(index="year", columns="month", values=var, aggfunc="mean")
    if hm.isnull().all().all():
        st.warning("Not enough data for heatmap.")
    else:
        heat = px.imshow(hm, labels=dict(x="Month", y="Year", color=var), color_continuous_scale="Viridis")
        st.plotly_chart(heat, use_container_width=True)

# === Extremes page ===
if view == "Extremes":
    st.header("Extreme events detection")
    df_e = apply_filters(df_daily)
    z_by = st.selectbox("Compute z-score per", options=["station_id","country"])
    extremes_df = detect_extremes(df_e, var, by=z_by, z_thresh=2.0)
    st.write(f"Found {len(extremes_df)} extreme records (|z|>=2) by {z_by}.")
    st.dataframe(extremes_df.sort_values("date", ascending=False).head(200))
    csv = extremes_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download extremes CSV", csv, "extremes.csv", "text/csv")

# === Compare page ===
if view == "Compare":
    st.header("Country comparison & boxplots")
    df_c = apply_filters(df_monthly)
    country_avg = df_c.groupby("country")[var].mean().reset_index().sort_values(var, ascending=False)
    st.bar_chart(country_avg.set_index("country")[var].head(20))
    st.subheader("Distribution among top countries")
    top_n = st.slider("Top N countries", 3, 20, 8)
    top_countries = df_c["country"].value_counts().nlargest(top_n).index.tolist()
    box = px.box(df_c[df_c["country"].isin(top_countries)], x="country", y=var)
    st.plotly_chart(box, use_container_width=True)

# === Station Clustering page ===
if view == "Station Clustering":
    st.header("Station clustering (KMeans)")
    df_k = df_daily.dropna(subset=["lat","lon",var])
    n_clusters = st.slider("Number of clusters (K)", 2, 20, 6)
    cluster_df = cluster_stations(df_k, var, n_clusters=n_clusters)
    st.dataframe(cluster_df[["station_id","country","lat","lon","cluster"]].drop_duplicates().head(200))
    # map show cluster
    fig_cluster = px.scatter_mapbox(cluster_df.sample(min(5000, len(cluster_df))), lat="lat", lon="lon", color="cluster", hover_name="station_id")
    fig_cluster.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_cluster, use_container_width=True)

# === Forecasting page ===
if view == "Forecasting":
    st.header("Forecasting (SARIMAX fallback + rolling mean)")
    df_f = apply_filters(df_monthly)
    sel_country = st.selectbox("Forecast country", options=(countries or all_countries))
    df_fc = df_f[df_f["country"] == sel_country].groupby("date")[var].mean().asfreq("MS").fillna(method="ffill")
    horizon = st.slider("Forecast horizon (months)", 1, 24, 6)
    with st.spinner("Computing forecast..."):
        fc_df = simple_forecast_sarimax_or_rolling(df_fc, horizon=horizon)
    # plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_fc.index, y=df_fc.values, name="History"))
    fig.add_trace(go.Scatter(x=fc_df.index, y=fc_df["forecast"], name="Forecast"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Forecast table**")
    st.dataframe(fc_df)

# Footer
st.markdown("---")
st.caption(" dashboard — built with Streamlit, Plotly, scikit-learn, and statsmodels.")
