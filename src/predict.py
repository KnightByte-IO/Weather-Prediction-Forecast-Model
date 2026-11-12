"""Live predictions: fetch recent weather for a place, run the trained ML models.

The recent "observed" hours come from Open-Meteo's forecast API (past_days), which
also returns the official numerical weather forecast that we show as a benchmark.
"""
import time
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests

from config import FORECAST_URL, MODEL_DIR, TIMEZONE, TRAIN_CITIES
from features import HORIZONS, add_daily_features, hourly_to_daily
from hourly_features import (HOURLY_FEATURES, MAX_LEAD, make_samples,
                             prepare_hourly)

FORECAST_HOURLY = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation",
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm", "soil_moisture_3_to_9cm",
    "vapour_pressure_deficit", "precipitation_probability", "weather_code",
]
FORECAST_DAILY = ["precipitation_sum", "precipitation_probability_max",
                  "temperature_2m_max", "temperature_2m_min", "relative_humidity_2m_mean",
                  "wind_speed_10m_max", "weather_code"]
TREE_MODELS = ("RandomForest", "XGBoost", "LightGBM")
WEATHER_VARS = ("t_max", "t_min", "rh_mean", "ws_max")   # see train_weather.py


# ------------------------------------------------------------------ data
def fetch_weather(lat: float, lon: float):
    params = {
        "latitude": lat, "longitude": lon, "timezone": TIMEZONE,
        "hourly": ",".join(FORECAST_HOURLY), "daily": ",".join(FORECAST_DAILY),
        "past_days": 40, "forecast_days": 8,
    }
    for attempt in range(4):
        try:
            r = requests.get(FORECAST_URL, params=params, timeout=60)
            if r.status_code < 500:
                break
        except (requests.ConnectionError, requests.Timeout):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
    r.raise_for_status()
    js = r.json()
    h = pd.DataFrame(js["hourly"])
    h["time"] = pd.to_datetime(h["time"])
    # ERA5 (training data) has a 0-7 cm soil layer; build it from the forecast API's layers.
    h["soil_moisture_0_to_7cm"] = (1 * h["soil_moisture_0_to_1cm"] + 2 * h["soil_moisture_1_to_3cm"]
                                   + 4 * h["soil_moisture_3_to_9cm"]) / 7
    d = pd.DataFrame(js["daily"])
    d["time"] = pd.to_datetime(d["time"])
    return h, d, float(js.get("elevation", 0.0))


def nearest_train_city(lat, lon, available=None):
    """Closest training city (its climate profile is used for the climatology features)."""
    def dist(c):
        la, lo = TRAIN_CITIES[c]["lat"], TRAIN_CITIES[c]["lon"]
        p1, p2 = np.radians(lat), np.radians(la)
        a = (np.sin((p2 - p1) / 2) ** 2
             + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo - lon) / 2) ** 2)
        return 6371 * 2 * np.arcsin(np.sqrt(a))
    pool = [c for c in TRAIN_CITIES if available is None or c in available]
    best = min(pool, key=dist)
    return best, dist(best)


# ---------------------------------------------------------------- models
@lru_cache(maxsize=None)
def _load(path):
    return joblib.load(path)


@lru_cache(maxsize=None)
def _daily_clim():
    return pd.read_parquet(MODEL_DIR / "daily_climatology.parquet")


@lru_cache(maxsize=None)
def _weather_clim():
    path = MODEL_DIR / "weather_climatology.parquet"
    return pd.read_parquet(path) if path.exists() else None


@lru_cache(maxsize=None)
def _hourly_clim():
    return pd.read_parquet(MODEL_DIR / "hourly" / "hourly_climatology.parquet")


@lru_cache(maxsize=None)
def _lstm(name):
    """LSTM weights run with NumPy (models/lstm/*.npz), so the app does not need PyTorch."""
    path = MODEL_DIR / "lstm" / f"{name}_lstm.npz"
    if not path.exists():
        return None
    from lstm_numpy import NumpyLSTM
    m = NumpyLSTM(path)
    return m, m.meta


def _lstm_predict(name, seq_df, static):
    got = _lstm(name)
    if got is None:
        return None
    m, ck = got
    x = seq_df[ck["seq_cols"]].to_numpy("float32")[-ck["seq_len"]:]
    if len(x) < ck["seq_len"]:
        return None
    x = np.nan_to_num((x - ck["mu"]) / ck["sd"]).astype("float32")
    s = ((np.asarray(static, "float32") - ck["smu"]) / ck["ssd"]).astype("float32")
    logit, mm = m(x, s)
    return 1.0 / (1.0 + np.exp(-logit)), np.expm1(np.clip(mm, 0, 6.2))


# ----------------------------------------------------------- predictions
def predict_daily(hourly_obs: pd.DataFrame, lat, lon, elevation, clim_city):
    """ML forecast for the next 7 days. Day 1 = today."""
    daily = hourly_to_daily(hourly_obs)          # complete days only, last one = yesterday
    daily = add_daily_features(daily)
    daily["lat"], daily["lon"], daily["elevation"] = lat, lon, elevation
    last = daily.iloc[[-1]].copy()
    base_date = last["date"].iloc[0]
    clim = _daily_clim()
    clim = clim[clim["city"] == clim_city].set_index("doy")
    wclim = _weather_clim()
    if wclim is not None:
        wclim = wclim[wclim["city"] == clim_city].set_index("doy")
    rows = []
    for h in HORIZONS:
        target = base_date + pd.Timedelta(days=h)
        doy = min(target.dayofyear, 365)
        last[f"clim_p_{h}"] = clim.loc[doy, "clim_p"]
        last[f"clim_mm_{h}"] = clim.loc[doy, "clim_mm"]
        probs, mms = {}, {}
        for name in TREE_MODELS:
            c = _load(MODEL_DIR / "daily" / f"clf_{name}_h{h}.joblib")
            probs[name] = float(c["model"].predict_proba(last[c["features"]])[:, 1][0])
            r = _load(MODEL_DIR / "daily" / f"reg_{name}_h{h}.joblib")
            mms[name] = max(0.0, float(r["model"].predict(last[r["features"]])[0]))
        weather = {}
        if wclim is not None:
            for v in WEATHER_VARS:
                last[f"clim_{v}_{h}"] = wclim.loc[doy, f"clim_{v}"]
            for v in WEATHER_VARS:
                path = MODEL_DIR / "weather" / f"{v}_h{h}.joblib"
                if path.exists():
                    w = _load(path)
                    weather[f"ml_{v}"] = float(w["model"].predict(last[w["features"]])[0])
        rows.append({"date": target, "h": h, "clim_p": last[f"clim_p_{h}"].iloc[0], **{
            f"p_{k}": v for k, v in probs.items()}, **{f"mm_{k}": v for k, v in mms.items()}, **weather})
    out = pd.DataFrame(rows)
    lstm = _lstm_predict("daily", daily, [lat, lon, elevation])
    if lstm is not None:
        out["p_LSTM"], out["mm_LSTM"] = lstm[0][: len(out)], lstm[1][: len(out)]
    pcols = [c for c in out if c.startswith("p_")]
    mcols = ["mm_XGBoost", "mm_LightGBM"] + (["mm_LSTM"] if "mm_LSTM" in out else [])
    out["ml_prob"] = out[pcols].mean(axis=1)
    out["ml_mm"] = out[mcols].mean(axis=1)
    # Show mm only where rain is reasonably likely, as forecasters do
    thr = _load(MODEL_DIR / "daily" / "ensemble_h1.joblib")["threshold"]
    out["ml_rain"] = out["ml_prob"] >= thr
    return out, base_date


def predict_hourly(hourly_obs: pd.DataFrame, lat, lon, elevation, clim_city):
    """ML forecast for each of the next 48 hours."""
    h = hourly_obs.copy()
    h["city"] = clim_city
    hp = prepare_hourly(h)
    hp["city"] = clim_city
    # make_samples looks up target rows, so pad the future 48 hours with empty rows
    origin_time = hp["time"].iloc[-1]
    future = pd.DataFrame({"time": pd.date_range(origin_time + pd.Timedelta(hours=1), periods=MAX_LEAD, freq="h")})
    future["city"] = clim_city
    hp2 = pd.concat([hp, future], ignore_index=True)
    mask = np.zeros(len(hp2), bool)
    mask[len(hp) - 1] = True
    s = make_samples(hp2, mask, range(1, MAX_LEAD + 1), _hourly_clim())
    s["lat"], s["lon"], s["elevation"] = lat, lon, elevation
    clf = _load(MODEL_DIR / "hourly" / "hourly_clf_LightGBM.joblib")
    reg = _load(MODEL_DIR / "hourly" / "hourly_reg_LightGBM.joblib")
    X = s[HOURLY_FEATURES].astype("float32")
    out = pd.DataFrame({"time": s["target_time"], "lead": s["lead"],
                        "p_LightGBM": clf["model"].predict_proba(X)[:, 1],
                        "mm_LightGBM": np.clip(reg["model"].predict(X), 0, None)})
    lstm = _lstm_predict("hourly", hp, [lat, lon, elevation])
    if lstm is not None:
        out["p_LSTM"], out["mm_LSTM"] = lstm[0], lstm[1]
    pcols = [c for c in out if c.startswith("p_")]
    mcols = [c for c in out if c.startswith("mm_")]
    out["ml_prob"] = out[pcols].mean(axis=1)
    out["ml_mm"] = out[mcols].mean(axis=1)
    return out, origin_time


def forecast_for(lat: float, lon: float):
    hourly, daily_official, elevation = fetch_weather(lat, lon)
    now = datetime.now(ZoneInfo(TIMEZONE)).replace(tzinfo=None, minute=0, second=0, microsecond=0)
    obs = hourly[hourly["time"] <= now].copy()
    official_hourly = hourly[hourly["time"] > now].copy()
    clim_city, clim_km = nearest_train_city(lat, lon, set(_daily_clim()["city"]))

    ml_daily, base_date = predict_daily(obs, lat, lon, elevation, clim_city)
    ml_hourly, origin = predict_hourly(obs, lat, lon, elevation, clim_city)

    official_daily = daily_official.rename(columns={"time": "date"})
    ml_daily = ml_daily.merge(official_daily, on="date", how="left")
    ml_hourly = ml_hourly.merge(official_hourly, on="time", how="left")
    return {
        "daily": ml_daily, "hourly": ml_hourly, "elevation": elevation,
        "climatology_city": clim_city, "climatology_km": clim_km,
        "data_until": origin, "now": now,
    }
