# src/utils.py
from pathlib import Path
import pandas as pd
import numpy as np
import pycountry
from functools import lru_cache
from sklearn.cluster import KMeans
import warnings

# caching wrapper for Streamlit caching but safe offline
def load_parquet_cached(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    return pd.read_parquet(path)

def ensure_iso3(df: pd.DataFrame) -> dict:
    """Return mapping country_name->iso3 where possible (pycountry)."""
    mapping = {}
    for name in df['country'].dropna().unique():
        try:
            c = pycountry.countries.lookup(name)
            mapping[name] = c.alpha_3
        except Exception:
            mapping[name] = None
    return mapping

def aggregate_time(df: pd.DataFrame, var: str, level: str = "Monthly") -> pd.DataFrame:
    if level == "Daily":
        return df.groupby("date")[var].mean().reset_index()
    if level == "Monthly":
        df2 = df.copy()
        df2["month"] = df2["date"].dt.to_period("M").dt.to_timestamp()
        return df2.groupby("month")[var].mean().reset_index().rename(columns={"month":"date"})
    if level == "Yearly":
        df2 = df.copy()
        df2["year"] = df2["date"].dt.year
        return df2.groupby("year")[var].mean().reset_index().rename(columns={"year":"date"})
    return df

def detect_extremes(df: pd.DataFrame, var: str, by: str = "station_id", z_thresh: float = 2.0) -> pd.DataFrame:
    df2 = df.copy()
    df2 = df2.dropna(subset=[var])
    df2['z'] = df2.groupby(by)[var].transform(lambda x: (x - x.mean())/x.std(ddof=0))
    return df2[df2['z'].abs() >= z_thresh].sort_values('z', key=lambda s: s.abs(), ascending=False)

def compute_kpis(df: pd.DataFrame, var: str) -> dict:
    s = df[var].dropna()
    return {
        "mean": float(s.mean()) if not s.empty else np.nan,
        "median": float(s.median()) if not s.empty else np.nan,
        "std": float(s.std()) if not s.empty else np.nan,
        "max": float(s.max()) if not s.empty else np.nan,
    }

def cluster_stations(df: pd.DataFrame, var: str, n_clusters: int = 6) -> pd.DataFrame:
    df2 = df.dropna(subset=["lat","lon",var]).copy()
    X = df2[["lat","lon",var]].values
    km = KMeans(n_clusters=n_clusters, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        labels = km.fit_predict(X)
    df2["cluster"] = labels
    return df2

# forecasting helper: try SARIMAX, fallback to rolling mean
def simple_forecast_sarimax_or_rolling(series: pd.Series, horizon: int = 6):
    """
    series: pd.Series indexed by datetime (monthly freq is ideal)
    returns: DataFrame index future dates with forecast column
    """
    import pandas as pd
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        s = series.dropna()
        if len(s) < 24:
            # too short -> fallback
            raise Exception("Too short for SARIMAX")
        model = SARIMAX(s, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=50)
        pred = res.get_forecast(steps=horizon)
        idx = pd.date_range(start=s.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
        return pd.DataFrame({"forecast": pred.predicted_mean}, index=idx)
    except Exception:
        # rolling mean fallback
        last = series.dropna().iloc[-1]
        idx = pd.date_range(start=series.dropna().index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
        forecast = pd.Series([series.dropna().rolling(3, min_periods=1).mean().iloc[-1]]*horizon, index=idx)
        return pd.DataFrame({"forecast": forecast})
