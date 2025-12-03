# app/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- CONFIG ---
# Relative path to parquet file
PARQUET_PATH = Path(__file__).parent.parent / "data" / "processed" / "monthly.parquet"

st.set_page_config(layout="wide", page_title="ClimateScope Dashboard")

# --- LOAD DATA ---
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error(f"Parquet file not found: {path}")
        st.stop()
    df = pd.read_parquet(path)
    
    # Ensure datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    elif 'month' in df.columns:
        df['date'] = pd.to_datetime(df['month'], errors='coerce')
    else:
        df['date'] = pd.NaT
    
    # Clean country names
    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.strip()
    else:
        df['country'] = 'Unknown'
    
    return df

df = load_data(PARQUET_PATH)

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filters")
country_list = sorted(df['country'].unique())
country = st.sidebar.selectbox("Select Country", country_list)

variables = ['temperature', 'precipitation', 'wind_speed', 'humidity']
var = st.sidebar.selectbox("Select Variable", variables)

# Date range filter
date_min = df['date'].min()
date_max = df['date'].max()
start_date, end_date = st.sidebar.date_input("Date Range", value=(date_min, date_max))

mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
df_filtered = df.loc[mask]

# Filter by country
df_country = df_filtered[df_filtered['country'] == country]

st.title(f"ClimateScope Dashboard — {country}")

# --- KPIs ---
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Temperature (°C)", f"{df_country['temperature'].mean():.2f}")
col2.metric("Total Precipitation (mm)", f"{df_country['precipitation'].sum():.2f}")
col3.metric("Avg Wind Speed (m/s)", f"{df_country['wind_speed'].mean():.2f}")
col4.metric("Avg Humidity (%)", f"{df_country['humidity'].mean():.2f}")

# --- Time Series Plot ---
st.subheader(f"{var.capitalize()} Over Time")
ts = df_country.groupby('date')[var].mean().reset_index()
fig_ts = px.line(ts, x='date', y=var, title=f"{var.capitalize()} Trend in {country}")
st.plotly_chart(fig_ts, use_container_width=True)

# --- Seasonal Heatmap ---
st.subheader(f"Seasonal Heatmap — {var.capitalize()}")
df_country['month_num'] = df_country['date'].dt.month
df_country['year'] = df_country['date'].dt.year
pivot = df_country.pivot_table(index='year', columns='month_num', values=var, aggfunc='mean')
fig_heat = px.imshow(pivot, labels=dict(x="Month", y="Year", color=var.capitalize()),
                     x=list(range(1,13)), y=pivot.index,
                     color_continuous_scale='Viridis')
st.plotly_chart(fig_heat, use_container_width=True)

# --- Country Comparison ---
st.subheader(f"Compare {var.capitalize()} Across Countries")
country_avg = df_filtered.groupby('country')[var].mean().reset_index().sort_values(var, ascending=False)
fig_bar = px.bar(country_avg, x='country', y=var, title=f"Average {var.capitalize()} by Country",
                 color=var, color_continuous_scale='Turbo')
st.plotly_chart(fig_bar, use_container_width=True)

# --- Extreme Events Table ---
st.subheader("Extreme Weather Events")
for metric in variables:
    df_filtered[f'{metric}_z'] = df_filtered.groupby('country')[metric].transform(
        lambda x: (x - x.mean())/x.std()
    )
extreme_df = df_filtered[(df_filtered[[f"{metric}_z" for metric in variables]].abs() >= 2).any(axis=1)]
st.dataframe(extreme_df[['date','country','station_id','temperature','precipitation','wind_speed','humidity']].sort_values('date', ascending=False))

# --- Map Visualization ---
st.subheader(f"{var.capitalize()} Map View")
fig_map = px.scatter_mapbox(df_country, lat="lat", lon="lon", color=var, size=var,
                            hover_name="station_id", hover_data=["temperature","precipitation","wind_speed","humidity"],
                            color_continuous_scale="Turbo", size_max=15, zoom=2)
fig_map.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig_map, use_container_width=True)

# --- Correlation Matrix ---
st.subheader("Correlation Between Variables")
corr = df_country[['temperature','precipitation','wind_speed','humidity']].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis')
st.plotly_chart(fig_corr, use_container_width=True)

# Footer info
st.info("Interactive ClimateScope Dashboard | Explore trends, extremes, seasonal patterns, and regional variations.")