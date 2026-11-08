"""Turn hourly weather into daily ML features and multi-day rain targets.

Each row = one city on one day D (a complete calendar day, IST).
Features describe day D and the days before it.
Targets describe days D+1 ... D+7 (h=1 is "today" when D is yesterday).
"""
import numpy as np
import pandas as pd

from config import RAIN_THRESHOLD_MM

HORIZONS = range(1, 8)


def _wind_uv(speed, direction_deg):
    rad = np.deg2rad(direction_deg)
    # Meteorological convention: direction is where wind comes FROM.
    return -speed * np.sin(rad), -speed * np.cos(rad)


def hourly_to_daily(h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one city's hourly frame to daily values."""
    h = h.copy()
    h["date"] = h["time"].dt.floor("D")
    h["hour"] = h["time"].dt.hour
    h["u"], h["v"] = _wind_uv(h["wind_speed_10m"], h["wind_direction_10m"])
    h["wet_hour"] = (h["precipitation"] >= 0.1).astype(float)
    eve = h[h["hour"] >= 18]

    g = h.groupby("date")
    d = pd.DataFrame({
        "precip_sum": g["precipitation"].sum(),
        "precip_hours": g["wet_hour"].sum(),
        "precip_max_1h": g["precipitation"].max(),
        "t_max": g["temperature_2m"].max(),
        "t_min": g["temperature_2m"].min(),
        "t_mean": g["temperature_2m"].mean(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "rh_max": g["relative_humidity_2m"].max(),
        "rh_min": g["relative_humidity_2m"].min(),
        "td_mean": g["dew_point_2m"].mean(),
        "p_msl_mean": g["pressure_msl"].mean(),
        "p_msl_min": g["pressure_msl"].min(),
        "p_msl_max": g["pressure_msl"].max(),
        "p_msl_last": g["pressure_msl"].last(),
        "sp_mean": g["surface_pressure"].mean(),
        "cc_mean": g["cloud_cover"].mean(),
        "cc_low_mean": g["cloud_cover_low"].mean(),
        "ws_mean": g["wind_speed_10m"].mean(),
        "ws_max": g["wind_speed_10m"].max(),
        "gust_max": g["wind_gusts_10m"].max(),
        "u_mean": g["u"].mean(),
        "v_mean": g["v"].mean(),
        "rad_sum": g["shortwave_radiation"].sum() / 1000.0,  # kWh/m2-ish scale
        "sm_mean": g["soil_moisture_0_to_7cm"].mean(),
        "vpd_mean": g["vapour_pressure_deficit"].mean(),
    })
    ge = eve.groupby("date")
    d["rh_eve"] = ge["relative_humidity_2m"].mean()
    d["cc_eve"] = ge["cloud_cover"].mean()
    d["precip_eve"] = ge["precipitation"].sum()
    d["p_msl_eve_trend"] = ge["pressure_msl"].last() - ge["pressure_msl"].first()

    # Keep only complete days (24 hourly values)
    counts = g["precipitation"].count()
    d = d[counts == 24]
    d.index.name = "date"
    return d.reset_index()


def add_daily_features(d: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling and calendar features for one city's daily frame (sorted by date)."""
    d = d.sort_values("date").reset_index(drop=True)
    # Reindex to a continuous calendar so shifts mean "days", not "rows"
    d = d.set_index("date").asfreq("D").reset_index()

    d["t_range"] = d["t_max"] - d["t_min"]
    d["dew_dep"] = d["t_mean"] - d["td_mean"]
    d["rainy"] = (d["precip_sum"] >= RAIN_THRESHOLD_MM).astype(float)
    d.loc[d["precip_sum"].isna(), "rainy"] = np.nan

    for k in (1, 2, 3):
        d[f"precip_lag{k}"] = d["precip_sum"].shift(k)
    for w in (3, 7, 14, 30):
        d[f"precip_roll{w}"] = d["precip_sum"].rolling(w, min_periods=max(1, w - 2)).sum()
    for w in (7, 30):
        d[f"rainy_days{w}"] = d["rainy"].rolling(w, min_periods=max(1, w - 2)).sum()
    d["p_change_1d"] = d["p_msl_mean"] - d["p_msl_mean"].shift(1)
    d["p_change_3d"] = d["p_msl_mean"] - d["p_msl_mean"].shift(3)
    d["rh_change_1d"] = d["rh_mean"] - d["rh_mean"].shift(1)
    d["t_change_1d"] = d["t_mean"] - d["t_mean"].shift(1)
    d["sm_change_3d"] = d["sm_mean"] - d["sm_mean"].shift(3)
    d["rh_roll7"] = d["rh_mean"].rolling(7, min_periods=5).mean()
    d["cc_roll7"] = d["cc_mean"].rolling(7, min_periods=5).mean()

    doy = d["date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["month"] = d["date"].dt.month
    return d


def add_targets(d: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        mm = d["precip_sum"].shift(-h)
        d[f"y_mm_{h}"] = mm
        d[f"y_rain_{h}"] = (mm >= RAIN_THRESHOLD_MM).astype(float).where(mm.notna())
        # Day-of-year of the target day, used for climatology lookups
        d[f"doy_{h}"] = (d["date"] + pd.Timedelta(days=h)).dt.dayofyear
    return d


def climatology_table(daily: pd.DataFrame, years=None, window: int = 7) -> pd.DataFrame:
    """Per-city rain probability and mean rain for each day of year (smoothed +/- window days).

    `years` restricts which years are used (pass training years only to avoid leakage).
    """
    df = daily[["city", "date", "precip_sum"]].dropna().copy()
    if years is not None:
        df = df[df["date"].dt.year.isin(years)]
    df["doy"] = df["date"].dt.dayofyear.clip(upper=365)
    df["rainy"] = (df["precip_sum"] >= RAIN_THRESHOLD_MM).astype(float)
    out = []
    for city, g in df.groupby("city"):
        by = g.groupby("doy").agg(p=("rainy", "mean"), mm=("precip_sum", "mean")).reindex(range(1, 366))
        by = by.interpolate(limit_direction="both")
        # circular smoothing
        ext = pd.concat([by.iloc[-window:], by, by.iloc[:window]])
        sm = ext.rolling(2 * window + 1, center=True).mean().iloc[window:-window]
        sm["city"] = city
        sm.index.name = "doy"
        out.append(sm.reset_index())
    return pd.concat(out, ignore_index=True).rename(columns={"p": "clim_p", "mm": "clim_mm"})


def attach_climatology(d: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        key = d[["city"]].copy()
        key["doy"] = d[f"doy_{h}"].clip(upper=365)
        m = key.merge(clim, on=["city", "doy"], how="left")
        d[f"clim_p_{h}"] = m["clim_p"].values
        d[f"clim_mm_{h}"] = m["clim_mm"].values
    return d


BASE_FEATURES = [
    "lat", "lon", "elevation",
    "precip_sum", "precip_hours", "precip_max_1h", "precip_eve",
    "t_max", "t_min", "t_mean", "t_range",
    "rh_mean", "rh_max", "rh_min", "rh_eve", "td_mean", "dew_dep",
    "p_msl_mean", "p_msl_min", "p_msl_max", "p_msl_last", "p_msl_eve_trend", "sp_mean",
    "cc_mean", "cc_low_mean", "cc_eve",
    "ws_mean", "ws_max", "gust_max", "u_mean", "v_mean",
    "rad_sum", "sm_mean", "vpd_mean",
    "precip_lag1", "precip_lag2", "precip_lag3",
    "precip_roll3", "precip_roll7", "precip_roll14", "precip_roll30",
    "rainy_days7", "rainy_days30",
    "p_change_1d", "p_change_3d", "rh_change_1d", "t_change_1d", "sm_change_3d",
    "rh_roll7", "cc_roll7",
    "doy_sin", "doy_cos", "month",
]


def features_for_horizon(h: int):
    return BASE_FEATURES + [f"clim_p_{h}", f"clim_mm_{h}"]
