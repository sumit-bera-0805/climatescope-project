import streamlit as st
import pandas as pd
import plotly.express as px
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
# SUNSET PALETTE CSS
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF3E0;
        color: #4E342E;
    }
    .stSidebar {
        background-color: #FFB74D;
        color: #4E342E;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4E342E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# HELPER — FORCE BLACK AXES & TITLES
# =====================================================
def apply_black_theme(fig):
    fig.update_layout(
        title_font=dict(color="black"),
        xaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        ),
        yaxis=dict(
            title_font=dict(color="black"),
            tickfont=dict(color="black")
        ),
        legend=dict(font=dict(color="black"))
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
    return (
        pd.read_parquet(DATA_DIR / "daily.parquet"),
        pd.read_parquet(DATA_DIR / "monthly.parquet")
    )

@st.cache_data
def load_reports():
    return (
        pd.read_csv(REPORTS_DIR / "country_mean_temperature.csv"),
        pd.read_csv(REPORTS_DIR / "global_correlation.csv")
    )

daily_df, monthly_df = load_core_data()
country_mean_temp, corr_df = load_reports()
daily_df["date"] = pd.to_datetime(daily_df["date"])

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🌍 ClimateScope")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Trends & Seasonality",
        "Air Quality Insights",
        "Spatial Analysis",
        "Correlation & Comparison"
    ]
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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🌡 Avg Temp (°C)", f"{filtered_daily.temperature.mean():.2f}")
    c2.metric("🌧 Total Precip (mm)", f"{filtered_daily.precipitation.sum():.2f}")
    c3.metric("💨 Avg Wind (m/s)", f"{filtered_daily.wind_speed.mean():.2f}")
    c4.metric("💧 Avg Humidity (%)", f"{filtered_daily.humidity.mean():.2f}")

    col1, col2 = st.columns(2)

    country_contrib = filtered_daily.groupby("country")[variable].mean().reset_index()
    fig_donut = px.pie(
        country_contrib,
        names="country",
        values=variable,
        hole=0.5,
        title="Country-wise Average Climate Contribution",
        color_discrete_sequence=["#FF7043", "#FFB74D", "#8E24AA"]
    )
    col1.plotly_chart(apply_black_theme(fig_donut), use_container_width=True)

    top_stations = (
        filtered_daily.groupby("station_id")[variable]
        .mean().reset_index().nlargest(5, variable)
    )
    fig_bar = px.bar(
        top_stations,
        x=variable,
        y="station_id",
        orientation="h",
        title="Top 5 Stations",
        color_discrete_sequence=["#FF7043"]
    )
    col2.plotly_chart(apply_black_theme(fig_bar), use_container_width=True)

# =====================================================
# PAGE 2 — TRENDS & SEASONALITY
# =====================================================
elif page == "Trends & Seasonality":
    st.title("📈 Trends & Seasonality")

    fig_line = px.line(
        filtered_daily,
        x="date",
        y=variable,
        color="country",
        title=f"{variable.title()} Trend",
        color_discrete_sequence=["#FF7043", "#FFB74D", "#8E24AA"]
    )
    st.plotly_chart(apply_black_theme(fig_line), use_container_width=True)

    pivot = monthly_df[monthly_df.country.isin(countries)].pivot_table(
        index="month", columns="country", values=variable, aggfunc="mean"
    )
    fig_heat = px.imshow(
        pivot.T,
        title="Seasonal Pattern Heatmap",
        color_continuous_scale=["#FFB74D", "#FF7043", "#8E24AA"]
    )
    st.plotly_chart(apply_black_theme(fig_heat), use_container_width=True)

# =====================================================
# PAGE 3 — AIR QUALITY INSIGHTS
# =====================================================
elif page == "Air Quality Insights":
    st.title("🌫 Air Quality Insights")

    fig_aqi = px.line(
        filtered_daily,
        x="date",
        y="air_quality_us-epa-index",
        color="country",
        title="Air Quality Index (US EPA)",
        color_discrete_sequence=["#FF7043", "#FFB74D", "#8E24AA"]
    )
    st.plotly_chart(apply_black_theme(fig_aqi), use_container_width=True)

    pollutants = [
        "air_quality_carbon_monoxide",
        "air_quality_ozone",
        "air_quality_nitrogen_dioxide",
        "air_quality_sulphur_dioxide",
        "air_quality_pm2.5",
        "air_quality_pm10"
    ]

    pollutant_avg = filtered_daily[pollutants].mean().reset_index()
    pollutant_avg.columns = ["Pollutant", "Average Value"]

    fig_pollution = px.bar(
        pollutant_avg,
        x="Pollutant",
        y="Average Value",
        title="Average Pollutant Levels",
        color_discrete_sequence=["#FF7043"]
    )
    st.plotly_chart(apply_black_theme(fig_pollution), use_container_width=True)

# =====================================================
# PAGE 4 — SPATIAL ANALYSIS
# =====================================================
elif page == "Spatial Analysis":
    st.title("🗺 Spatial Climate Analysis")

    st.subheader("🌍 Global Mean Temperature (Country Level)")
    html_map = FIGURES_DIR / "choropleth_mean_temp.html"
    if html_map.exists():
        components.html(html_map.read_text(encoding="utf-8"), height=550)
    else:
        st.warning("Choropleth file not found")

    station_avg = (
        filtered_daily.groupby(["station_id", "country", "lat", "lon"])
        .mean(numeric_only=True).reset_index()
    )

    fig_map = px.scatter_mapbox(
        station_avg,
        lat="lat",
        lon="lon",
        color=variable,
        size=variable,
        mapbox_style="carto-positron",
        zoom=2,
        title="Station-Level Climate Distribution",
        color_continuous_scale=["#FFB74D", "#FF7043", "#8E24AA"]
    )
    st.plotly_chart(apply_black_theme(fig_map), use_container_width=True)

# =====================================================
# PAGE 5 — CORRELATION & COMPARISON
# =====================================================
elif page == "Correlation & Comparison":
    st.title("📊 Correlation & Comparison")

    fig_corr = px.imshow(
        corr_df.drop(columns=["Unnamed: 0"], errors="ignore"),
        text_auto=True,
        title="Climate Variable Correlation Matrix",
        color_continuous_scale=["#FFB74D", "#FF7043", "#8E24AA"]
    )
    st.plotly_chart(apply_black_theme(fig_corr), use_container_width=True)

    fig_country = px.bar(
        country_mean_temp.sort_values("temperature"),
        x="temperature",
        y="country",
        orientation="h",
        title="Average Temperature by Country",
        color_discrete_sequence=["#FF7043"]
    )
    st.plotly_chart(apply_black_theme(fig_country), use_container_width=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    ---
    **ClimateScope Dashboard**  
    Interactive climate analytics platform
    """,
    unsafe_allow_html=True
)
