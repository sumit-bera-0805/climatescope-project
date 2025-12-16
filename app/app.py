import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit.components.v1 as components

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ClimateScope Dashboard",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# COLOR PALETTE (CENTRALIZED)
# =====================================================
CATEGORICAL_COLORS = [
    "#1E88E5", "#00897B", "#3949AB", "#7CB342", "#D84315", "#546E7A"
]

SEQUENTIAL_SCALE = [
    "#E3F2FD", "#90CAF9", "#42A5F5", "#FB8C00", "#E53935"
]

DIVERGING_SCALE = [
    "#283593", "#5C6BC0", "#F5F5F5", "#FF8A65", "#BF360C"
]

# =====================================================
# SUNSET UI THEME (CSS)
# =====================================================
st.markdown("""
<style>
.stApp {background-color: #FFF3E0; color: #4E342E;}
section[data-testid="stSidebar"] {background-color: #FFE0B2; color: #4E342E;}
h1,h2,h3,h4,h5,h6 {color: #4E342E;}
div[data-testid="stMetric"] {background-color: #FFE0B2; border-radius: 12px; padding: 12px; border: 1px solid #D7CCC8; box-shadow: 3px 3px 6px #BDBDBD;}
div[data-testid="stMetricLabel"] {color: #6D4C41;}
div[data-testid="stMetricValue"] {color: #FF7043; font-weight: 600;}
label, span {color: #4E342E !important;}
hr {border-color: #D7CCC8;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HELPER — FORCE BLACK AXES & TITLES
# =====================================================
def apply_black_theme(fig):
    fig.update_layout(
        title_font=dict(color="black"),
        xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
        legend=dict(font=dict(color="black")),
        plot_bgcolor="#FFF3E0",
        paper_bgcolor="#FFF3E0"
    )
    return fig

# =====================================================
# PATH CONFIG
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_core_data():
    return pd.read_parquet(DATA_DIR / "daily.parquet"), pd.read_parquet(DATA_DIR / "monthly.parquet")

@st.cache_data
def load_reports():
    return pd.read_csv(REPORTS_DIR / "country_mean_temperature.csv"), pd.read_csv(REPORTS_DIR / "global_correlation.csv")

daily_df, monthly_df = load_core_data()
country_mean_temp, corr_df = load_reports()
daily_df["date"] = pd.to_datetime(daily_df["date"])

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌍 ClimateScope")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Trends & Seasonality", "Extreme Events", "Maps", "Distribution & Statistical Views"]
)

countries = st.sidebar.multiselect(
    "Select Country/Countries",
    sorted(daily_df["country"].unique()),
    default=[daily_df["country"].unique()[0]]
)

variable = st.sidebar.selectbox(
    "Select Variable",
    ["temperature", "precipitation", "wind_speed", "humidity"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    [daily_df["date"].min(), daily_df["date"].max()]
)

filtered_daily = daily_df[
    (daily_df["country"].isin(countries)) &
    (daily_df["date"] >= pd.to_datetime(date_range[0])) &
    (daily_df["date"] <= pd.to_datetime(date_range[1]))
]

# =====================================================
# PAGE 1 — OVERVIEW
# =====================================================
if page == "Overview":
    st.title("🌦 ClimateScope Dashboard")
    st.caption("High-level climate insights across regions and stations")

    # KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡 Avg Temp (°C)", f"{filtered_daily.temperature.mean():.2f}")
    c2.metric("🌧 Total Precip (mm)", f"{filtered_daily.precipitation.sum():.2f}")
    c3.metric("💨 Avg Wind (m/s)", f"{filtered_daily.wind_speed.mean():.2f}")
    c4.metric("💧 Avg Humidity (%)", f"{filtered_daily.humidity.mean():.2f}")

    # Layout: Gauge + Wind Polar Plot side by side
    col1, col2 = st.columns([1, 1])

    # Gauge for Average Precipitation Probability (new)
    with col1:
        if "precipitation_probability" in filtered_daily.columns:
            avg_precip_prob = filtered_daily["precipitation_probability"].mean()
        else:
            avg_precip_prob = (filtered_daily.precipitation > 0).mean() * 100  # fallback
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_precip_prob,
            title={"text": "🌧 Avg Precipitation Probability (%)"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#1E88E5"}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Wind Direction Polar Plot
    with col2:
        if "wind_direction" in filtered_daily.columns:
            wind_avg = filtered_daily.groupby("wind_direction")["wind_speed"].mean().reset_index()
            fig_wind = px.bar_polar(
                wind_avg, r="wind_speed", theta="wind_direction", color="wind_speed",
                color_continuous_scale=SEQUENTIAL_SCALE, title="💨 Average Wind Direction & Speed"
            )
            st.plotly_chart(fig_wind, use_container_width=True)

    # Humidity + Rain Probability Bar
    st.subheader("💧 Humidity & Rain Probability by Country")
    humidity_rain = filtered_daily.groupby("country").agg(
        avg_humidity=("humidity", "mean"),
        rain_prob=("precipitation", lambda x: (x > 0).mean() * 100)
    ).reset_index()
    
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=humidity_rain["country"],
        y=humidity_rain["avg_humidity"],
        name="Avg Humidity (%)",
        marker_color="#7CB342"
    ))
    fig_bar.add_trace(go.Bar(
        x=humidity_rain["country"],
        y=humidity_rain["rain_prob"],
        name="Rain Probability (%)",
        marker_color="#1E88E5"
    ))
    fig_bar.update_layout(
        barmode='group',
        xaxis_title="Country",
        yaxis_title="Percentage",
        title="Humidity & Rain Probability",
        plot_bgcolor="#FFF3E0",
        paper_bgcolor="#FFF3E0",
        legend=dict(font=dict(color="black"))
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# =====================================================
# PAGE 2 — TRENDS & SEASONALITY
# =====================================================
elif page == "Trends & Seasonality":
    st.title("📈 Trends & Seasonality")

    # Line chart daily/monthly/yearly
    fig_line = px.line(filtered_daily, x="date", y=variable, color="country",
                       title=f"{variable.title()} Trend", color_discrete_sequence=CATEGORICAL_COLORS)
    st.plotly_chart(apply_black_theme(fig_line), use_container_width=True)

    # Multi-line comparison (monthly)
    pivot = monthly_df[monthly_df.country.isin(countries)].pivot_table(
        index="month", columns="country", values=variable, aggfunc="mean"
    )
    fig_multi = px.line(pivot, x=pivot.index, y=pivot.columns,
                        labels={"value": variable.title(), "month": "Month"},
                        title=f"{variable.title()} Comparison Across Countries")
    st.plotly_chart(apply_black_theme(fig_multi), use_container_width=True)

    # Area chart cumulative
    pivot_cum = filtered_daily.groupby("date")[variable].sum().cumsum().reset_index()
    fig_area = px.area(pivot_cum, x="date", y=variable, title=f"Cumulative {variable.title()}")
    st.plotly_chart(apply_black_theme(fig_area), use_container_width=True)

# =====================================================
# PAGE 3 — EXTREME EVENTS
# =====================================================
elif page == "Extreme Events":
    st.title("⚠️ Extreme Weather & Anomaly Visualizations")
    st.caption("Interactive insights into extreme events and thresholds")

    # -------------------------------
    # Lollipop Chart: Extreme Days Overview
    # -------------------------------
    st.subheader("📍 Extreme Days Overview")
    # Define extreme threshold: e.g., 95th percentile
    extreme_threshold = filtered_daily[variable].quantile(0.95)
    extreme_days = filtered_daily[filtered_daily[variable] > extreme_threshold]
    extreme_count = extreme_days.groupby("country")[variable].count().reset_index()
    
    fig_lollipop = go.Figure()
    fig_lollipop.add_trace(go.Scatter(
        x=extreme_count[variable],
        y=extreme_count["country"],
        mode='markers+lines',
        marker=dict(size=12, color="#D84315"),
        line=dict(color="#D84315", width=2)
    ))
    fig_lollipop.update_layout(
        title=f"Extreme {variable.title()} Days per Country",
        xaxis_title=f"Number of Days (>{extreme_threshold:.2f})",
        yaxis_title="Country",
        plot_bgcolor="#FFF3E0",
        paper_bgcolor="#FFF3E0"
    )
    st.plotly_chart(fig_lollipop, use_container_width=True)

    # -------------------------------
    # Threshold Highlights
    # -------------------------------
    st.subheader("⚡ Threshold Highlights")
    threshold_df = filtered_daily.copy()
    threshold_df["is_extreme"] = threshold_df[variable] > extreme_threshold

    fig_threshold = px.scatter(
        threshold_df, x="date", y=variable, color="is_extreme",
        color_discrete_map={True: "#D84315", False: "#7CB342"},
        hover_data=["country"], title=f"{variable.title()} Threshold Highlight"
    )
    st.plotly_chart(apply_black_theme(fig_threshold), use_container_width=True)

    # -------------------------------
    # Alerts Summary
    # -------------------------------
    st.subheader("🚨 Alerts Summary")

    alert_counts = threshold_df.groupby("country")["is_extreme"].sum().reset_index()
    c1, c2, c3 = st.columns(3)
    if len(alert_counts) > 0:
        c1.metric("Top Country Alerts", f"{alert_counts['country'][alert_counts['is_extreme'].idxmax()]} ({alert_counts['is_extreme'].max()})")
        c2.metric("Total Alerts", f"{alert_counts['is_extreme'].sum()}")
        c3.metric("Avg Alerts per Country", f"{alert_counts['is_extreme'].mean():.2f}")
    else:
        c1.metric("Top Country Alerts", "N/A")
        c2.metric("Total Alerts", "0")
        c3.metric("Avg Alerts per Country", "0")

    # Interactive Scatter Plot: Extreme Events
    fig_alert_scatter = px.scatter(
        threshold_df[threshold_df["is_extreme"]],
        x="lon", y="lat",
        size=variable, color=variable,
        hover_name="country",
        color_continuous_scale=SEQUENTIAL_SCALE,
        title=f"Extreme {variable.title()} Events Scatter"
    )
    st.plotly_chart(apply_black_theme(fig_alert_scatter), use_container_width=True)

    # Interactive Bubble Chart: Extreme Events by Country
    extreme_country = threshold_df[threshold_df["is_extreme"]].groupby("country").agg(
        total_value=(variable, "sum"),
        event_count=(variable, "count")
    ).reset_index()
    fig_bubble = px.scatter(
        extreme_country, x="country", y="total_value", size="event_count",
        color="total_value", hover_name="country",
        color_continuous_scale=SEQUENTIAL_SCALE,
        title=f"Extreme {variable.title()} Bubble Chart by Country"
    )
    st.plotly_chart(apply_black_theme(fig_bubble), use_container_width=True)

# =====================================================
# PAGE 4 — SPATIAL ANALYSIS (NO SIDEBAR FILTERS)
# =====================================================
elif page == "Maps":
    st.title("🗺️ Spatial Climate Analysis")
    st.caption("Interactive spatial insights with station data, heatmaps, and extreme events")

    # --------------------------------
    # Use globally selected variable
    # --------------------------------
    spatial_variable = variable

    # --------------------------------
    # Prepare spatial dataframe
    # --------------------------------
    spatial_df = filtered_daily.copy()

    if spatial_df.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # --------------------------------
    # Normalize marker size (FIXED)
    # --------------------------------
    if spatial_variable in spatial_df.columns:
        min_val = spatial_df[spatial_variable].min()
        max_val = spatial_df[spatial_variable].max()

        size_values = (
            (spatial_df[spatial_variable] - min_val)
            / (max_val - min_val + 1e-6)
        ) * 25 + 5
    else:
        size_values = 10

    # --------------------------------
    # 📍 Station Map
    # --------------------------------
    st.subheader("📍 Station Map")

    fig_map = px.scatter_mapbox(
        spatial_df,
        lat="lat",
        lon="lon",
        color=spatial_variable,
        size=size_values,
        hover_name="station_id",
        hover_data={
            "country": True,
            "date": True,
            spatial_variable: True,
            "lat": False,
            "lon": False
        },
        color_continuous_scale=SEQUENTIAL_SCALE,
        zoom=2,
        mapbox_style="carto-positron",
        title=f"{spatial_variable.title()} at Weather Stations"
    )

    st.plotly_chart(apply_black_theme(fig_map), use_container_width=True)

    # --------------------------------
    # 🌍 Country-wise Average Choropleth
    # --------------------------------
    st.subheader(f"🌡 Country-wise Average {spatial_variable.title()}")

    country_avg_map = (
        spatial_df
        .groupby("country")[spatial_variable]
        .mean()
        .reset_index()
    )

    fig_choropleth = px.choropleth(
        country_avg_map,
        locations="country",
        locationmode="country names",
        color=spatial_variable,
        color_continuous_scale=SEQUENTIAL_SCALE,
        title=f"Average {spatial_variable.title()} by Country"
    )

    st.plotly_chart(apply_black_theme(fig_choropleth), use_container_width=True)

    # --------------------------------
    # 📊 Top & Bottom Stations
    # --------------------------------
    st.subheader("📊 Top & Extreme Locations")

    station_mean = (
        spatial_df
        .groupby("station_id")[spatial_variable]
        .mean()
        .reset_index()
    )

    top5_max = station_mean.nlargest(5, spatial_variable)
    top5_min = station_mean.nsmallest(5, spatial_variable)

    c1, c2 = st.columns(2)

    with c1:
        fig_top5 = px.bar(
            top5_max,
            x="station_id",
            y=spatial_variable,
            color=spatial_variable,
            color_continuous_scale=SEQUENTIAL_SCALE,
            title=f"Top 5 Highest {spatial_variable.title()}"
        )
        st.plotly_chart(apply_black_theme(fig_top5), use_container_width=True)

    with c2:
        fig_bottom5 = px.bar(
            top5_min,
            x="station_id",
            y=spatial_variable,
            color=spatial_variable,
            color_continuous_scale=SEQUENTIAL_SCALE,
            title=f"Top 5 Lowest {spatial_variable.title()}"
        )
        st.plotly_chart(apply_black_theme(fig_bottom5), use_container_width=True)

    # --------------------------------
    # ⚡ Extreme Event Markers (95th percentile)
    # --------------------------------
    st.subheader("⚡ Extreme Event Markers")

    extreme_threshold_map = spatial_df[spatial_variable].quantile(0.95)
    extreme_events = spatial_df[spatial_df[spatial_variable] > extreme_threshold_map]

    if extreme_events.empty:
        st.info("No extreme events detected for the selected variable.")
    else:
        extreme_size = size_values.loc[extreme_events.index]

        fig_extreme = px.scatter_mapbox(
            extreme_events,
            lat="lat",
            lon="lon",
            size=extreme_size,
            color=spatial_variable,
            hover_name="station_id",
            hover_data={
                "country": True,
                "date": True,
                spatial_variable: True
            },
            color_continuous_scale=DIVERGING_SCALE,
            zoom=2,
            mapbox_style="carto-positron",
            title=f"Extreme {spatial_variable.title()} Events (>95th Percentile)"
        )

        st.plotly_chart(apply_black_theme(fig_extreme), use_container_width=True)
# =====================================================
# PAGE 5 — DISTRIBUTION & STATISTICAL VIEWS
# =====================================================
elif page == "Distribution & Statistical Views":
    st.title("🌡️ Distribution & Statistical Views")
    st.caption("Explore spread, probability, percentiles, and uncertainty in climate data")

    # --------------------------------
    # Prepare clean data
    # --------------------------------
    dist_df = filtered_daily[[variable, "date", "country"]].dropna()

    # --------------------------------
    # 1️⃣ HISTOGRAM
    # --------------------------------
    #st.subheader("📊 Histogram")
    fig_hist = px.histogram(
        dist_df,
        x=variable,
        color="country",
        nbins=40,
        opacity=0.75,
        color_discrete_sequence=CATEGORICAL_COLORS,
        title=f"Histogram of {variable.title()}"
    )
    st.plotly_chart(apply_black_theme(fig_hist), use_container_width=True)

    # --------------------------------
    # 2️⃣ DENSITY PLOT (Smoothed Histogram)
    # --------------------------------
    #st.subheader("📈 Density Plot")
    fig_density = px.histogram(
        dist_df,
        x=variable,
        color="country",
        nbins=60,
        histnorm="probability density",
        opacity=0.6,
        marginal="rug",
        color_discrete_sequence=CATEGORICAL_COLORS,
        title=f"Density Distribution of {variable.title()}"
    )
    st.plotly_chart(apply_black_theme(fig_density), use_container_width=True)

    # --------------------------------
    # 3️⃣ CDF (Cumulative Distribution Function)
    # --------------------------------
    #st.subheader("📐 Cumulative Distribution Function (CDF)")
    fig_cdf = px.ecdf(
        dist_df,
        x=variable,
        color="country",
        color_discrete_sequence=CATEGORICAL_COLORS,
        title=f"CDF of {variable.title()}"
    )
    st.plotly_chart(apply_black_theme(fig_cdf), use_container_width=True)

    # --------------------------------
    # 4️⃣ PERCENTILE BANDS (25–75 & 10–90)
    # --------------------------------
    #st.subheader("📉 Percentile Bands")

    percentile_df = dist_df.groupby("date")[variable].agg(
        p10=lambda x: x.quantile(0.10),
        p25=lambda x: x.quantile(0.25),
        p50="median",
        p75=lambda x: x.quantile(0.75),
        p90=lambda x: x.quantile(0.90)
    ).reset_index()

    fig_percentile = go.Figure()

    # 10–90 band
    fig_percentile.add_trace(go.Scatter(
        x=percentile_df["date"],
        y=percentile_df["p90"],
        line=dict(width=0),
        showlegend=False
    ))
    fig_percentile.add_trace(go.Scatter(
        x=percentile_df["date"],
        y=percentile_df["p10"],
        fill="tonexty",
        fillcolor="rgba(255,112,67,0.2)",
        line=dict(width=0),
        name="10–90 Percentile"
    ))

    # 25–75 band
    fig_percentile.add_trace(go.Scatter(
        x=percentile_df["date"],
        y=percentile_df["p75"],
        line=dict(width=0),
        showlegend=False
    ))
    fig_percentile.add_trace(go.Scatter(
        x=percentile_df["date"],
        y=percentile_df["p25"],
        fill="tonexty",
        fillcolor="rgba(30,136,229,0.25)",
        line=dict(width=0),
        name="25–75 Percentile"
    ))

    # Median
    fig_percentile.add_trace(go.Scatter(
        x=percentile_df["date"],
        y=percentile_df["p50"],
        line=dict(color="black", width=2),
        name="Median"
    ))

    fig_percentile.update_layout(
        title=f"Percentile Bands of {variable.title()} Over Time",
        xaxis_title="Date",
        yaxis_title=variable.title()
    )
    st.plotly_chart(apply_black_theme(fig_percentile), use_container_width=True)

    # --------------------------------
    # 5️⃣ RANGE BANDS (MIN–MAX SHADING)
    # --------------------------------
    #st.subheader("📦 Range Bands (Min–Max)")

    range_df = dist_df.groupby("date")[variable].agg(
        min_val="min",
        max_val="max",
        mean_val="mean"
    ).reset_index()

    fig_range = go.Figure()

    fig_range.add_trace(go.Scatter(
        x=range_df["date"],
        y=range_df["max_val"],
        line=dict(width=0),
        showlegend=False
    ))
    fig_range.add_trace(go.Scatter(
        x=range_df["date"],
        y=range_df["min_val"],
        fill="tonexty",
        fillcolor="rgba(124,179,66,0.25)",
        line=dict(width=0),
        name="Min–Max Range"
    ))
    fig_range.add_trace(go.Scatter(
        x=range_df["date"],
        y=range_df["mean_val"],
        line=dict(color="#D84315", width=2),
        name="Mean"
    ))

    fig_range.update_layout(
        title=f"Min–Max Range of {variable.title()} Over Time",
        xaxis_title="Date",
        yaxis_title=variable.title()
    )
    st.plotly_chart(apply_black_theme(fig_range), use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown("""
---
**ClimateScope Dashboard**  
Interactive climate analytics platform
""", unsafe_allow_html=True)
