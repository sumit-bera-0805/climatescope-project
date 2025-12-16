# src/data_pipeline.py

import pandas as pd
import os

# -----------------------
# 1. Load raw dataset
# -----------------------
def load_raw(path):
    print("Loading raw dataset...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found!")
    df = pd.read_csv(path, parse_dates=['last_updated'], low_memory=False)
    return df

# -----------------------
# 2. Standardize column names
# -----------------------
def standardize(df):
    print("Standardizing column names...")
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # Rename columns for consistent use in pipeline
    df = df.rename(columns={
        'temperature_celsius': 'temperature',
        'wind_kph': 'wind_speed',
        'precip_mm': 'precipitation',
        'location_name': 'station_id',
        'latitude': 'lat',
        'longitude': 'lon',
        'last_updated': 'date'
    })
    return df

# -----------------------
# 3. Convert units
# -----------------------
def convert_units(df):
    print("Converting units...")

    # Wind kph → m/s
    if 'wind_speed' in df.columns:
        df['wind_speed'] = df['wind_speed'] * 0.27778

    return df

# -----------------------
# 4. Handle missing values
# -----------------------
def impute_stationwise(df, cols=['temperature', 'humidity']):
    print("Handling missing values...")

    if "station_id" not in df.columns:
        raise ValueError("Column 'station_id' is missing!")

    df = df.sort_values(['station_id', 'date'])

    for c in cols:
        if c in df.columns:
            # Use transform instead of apply to keep index aligned
            df[c] = df.groupby('station_id')[c].transform(
                lambda s: s.interpolate(limit_direction='both')
            )

    df = df.dropna(subset=['temperature', 'humidity'], how='any')
    return df

# -----------------------
# 5. Remove outliers
# -----------------------
def clean_outliers(df):
    print("Cleaning outliers...")

    df = df[(df['temperature'] > -100) & (df['temperature'] < 60)]

    if 'humidity' in df.columns:
        df = df[(df['humidity'] >= 0) & (df['humidity'] <= 100)]

    return df

# -----------------------
# 6. Aggregate monthly
# -----------------------
def aggregate_monthly(df):
    print("Aggregating to monthly level...")
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()

    agg = df.groupby(['station_id', 'country', 'month']).agg({
        'temperature': 'mean',
        'precipitation': 'sum',
        'wind_speed': 'mean',
        'humidity': 'mean',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()

    return agg

# -----------------------
# 7. Run full pipeline
# -----------------------
def run_all():
    print("Starting full preprocessing pipeline...")

    # Raw CSV path
    raw_path = r"C:\Users\mukhe\OneDrive\Desktop\coding\project1\climatescope-project\data\raw\GlobalWeatherRepository.csv"
    
    df = load_raw(raw_path)
    df = standardize(df)
    df = convert_units(df)
    df = impute_stationwise(df)
    df = clean_outliers(df)

    # Ensure processed folder exists
    processed_dir = r"C:\Users\mukhe\OneDrive\Desktop\coding\project1\climatescope-project\data\processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Save cleaned daily data
    daily_file = os.path.join(processed_dir, "daily.parquet")
    print("Saving cleaned daily data...")
    df.to_parquet(daily_file, index=False)

    # Save monthly aggregated data
    df_month = aggregate_monthly(df)
    monthly_file = os.path.join(processed_dir, "monthly.parquet")
    print("Saving monthly aggregated data...")
    df_month.to_parquet(monthly_file, index=False)

    print("Pipeline completed successfully!")

# -----------------------
# Execute pipeline
# -----------------------
if __name__ == "__main__":
    run_all()