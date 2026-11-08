"""Hourly features for the 48-hour "today / tomorrow" detail model.

A sample = (city, origin hour t, lead k). Features describe the weather at and
before hour t; the target is rain in hour t+k (k = 1..48).
"""
import numpy as np
import pandas as pd

WET_HOUR_MM = 0.2      # an hour counts as "rainy" at >= 0.2 mm
MAX_LEAD = 48

STATE_COLS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "cloud_cover", "cloud_cover_low", "wind_speed_10m", "wind_gusts_10m",
    "shortwave_radiation", "soil_moisture_0_to_7cm", "vapour_pressure_deficit",
    "u", "v",
]


def prepare_hourly(h: pd.DataFrame) -> pd.DataFrame:
    """Continuous hourly frame for one city with derived origin-time features."""
    h = h.sort_values("time").drop_duplicates("time").set_index("time").asfreq("h")
    rad = np.deg2rad(h["wind_direction_10m"])
    h["u"] = -h["wind_speed_10m"] * np.sin(rad)
    h["v"] = -h["wind_speed_10m"] * np.cos(rad)
    p = h["precipitation"]
    h["wet"] = (p >= WET_HOUR_MM).astype(float).where(p.notna())
    for w in (3, 6, 12, 24, 72, 168):
        h[f"precip_last{w}h"] = p.rolling(w, min_periods=1).sum()
    h["wet_hours_last24h"] = h["wet"].rolling(24, min_periods=1).sum()
    h["p_change_3h"] = h["pressure_msl"].diff(3)
    h["p_change_12h"] = h["pressure_msl"].diff(12)
    h["p_change_24h"] = h["pressure_msl"].diff(24)
    h["rh_change_3h"] = h["relative_humidity_2m"].diff(3)
    h["t_change_3h"] = h["temperature_2m"].diff(3)
    h["cc_change_3h"] = h["cloud_cover"].diff(3)
    h["dew_dep"] = h["temperature_2m"] - h["dew_point_2m"]
    return h.reset_index()


ORIGIN_FEATURES = STATE_COLS + [
    "precipitation", "precip_last3h", "precip_last6h", "precip_last12h",
    "precip_last24h", "precip_last72h", "precip_last168h", "wet_hours_last24h",
    "p_change_3h", "p_change_12h", "p_change_24h", "rh_change_3h",
    "t_change_3h", "cc_change_3h", "dew_dep",
]
HOURLY_FEATURES = (["lat", "lon", "elevation", "lead", "origin_hour_sin", "origin_hour_cos",
                    "target_hour_sin", "target_hour_cos", "doy_sin", "doy_cos", "clim_wet_p"]
                   + ORIGIN_FEATURES)


def hourly_climatology(hourly: pd.DataFrame, years) -> pd.DataFrame:
    """P(wet hour) by city x month x hour of day, from the given years only."""
    x = hourly[hourly["time"].dt.year.isin(years)].dropna(subset=["wet"])
    x = x.assign(month=x["time"].dt.month, hour=x["time"].dt.hour)
    return x.groupby(["city", "month", "hour"], as_index=False)["wet"].mean().rename(
        columns={"wet": "clim_wet_p"})


def make_samples(h: pd.DataFrame, origins_mask, leads, clim: pd.DataFrame, rng=None,
                 lead_fraction: float = 1.0) -> pd.DataFrame:
    """Stack (origin, lead) samples for one city's prepared hourly frame."""
    h = h.reset_index(drop=True)
    origin_idx = np.flatnonzero(origins_mask)
    blocks = []
    for k in leads:
        idx = origin_idx[origin_idx + k < len(h)]
        if lead_fraction < 1.0 and rng is not None:
            idx = idx[rng.random(len(idx)) < lead_fraction]
        o = h.loc[idx, ["city", "time"] + ORIGIN_FEATURES].reset_index(drop=True)
        tgt = h.loc[idx + k, ["time", "precipitation", "wet"]].reset_index(drop=True)
        o["lead"] = k
        o["target_time"] = tgt["time"].values
        o["y_mm"] = tgt["precipitation"].values
        o["y_wet"] = tgt["wet"].values
        blocks.append(o)
    s = pd.concat(blocks, ignore_index=True)
    oh, th = s["time"].dt.hour, s["target_time"].dt.hour
    s["origin_hour_sin"], s["origin_hour_cos"] = np.sin(2 * np.pi * oh / 24), np.cos(2 * np.pi * oh / 24)
    s["target_hour_sin"], s["target_hour_cos"] = np.sin(2 * np.pi * th / 24), np.cos(2 * np.pi * th / 24)
    doy = s["target_time"].dt.dayofyear
    s["doy_sin"], s["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    s["month"], s["hour"] = s["target_time"].dt.month, th
    s = s.merge(clim, on=["city", "month", "hour"], how="left").drop(columns=["month", "hour"])
    return s
