"""Train 7-day ML forecasts for other weather variables: max/min temperature, humidity, wind.

For each variable and each lead day h = 1..7 a LightGBM regressor is trained, and compared
with two baselines: persistence (same as the last observed day) and climatology
(the city's average for that day of the year, from training years only).

Usage:  python src/train_weather.py
Output: models/weather/*.joblib, models/weather_climatology.parquet,
        reports/weather_metrics.csv, reports/weather_test_predictions.parquet
"""
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from features import BASE_FEATURES, HORIZONS

warnings.filterwarnings("ignore", message=".*eval_set.*")
SEED = 42
TRAIN_YEARS = range(2014, 2023)

# variable -> (label, unit, tolerance used for the "within tolerance" accuracy)
TARGETS = {
    "t_max": ("Max temperature", "°C", 2.0),
    "t_min": ("Min temperature", "°C", 2.0),
    "rh_mean": ("Mean humidity", "%", 10.0),
    "ws_max": ("Max wind speed", "km/h", 5.0),
}


def weather_climatology(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """City x day-of-year mean of each target variable, smoothed, from training years only."""
    x = df[df["date"].dt.year.isin(TRAIN_YEARS)].copy()
    x["doy"] = x["date"].dt.dayofyear.clip(upper=365)
    out = []
    for city, g in x.groupby("city"):
        by = g.groupby("doy")[list(TARGETS)].mean().reindex(range(1, 366)).interpolate(limit_direction="both")
        ext = pd.concat([by.iloc[-window:], by, by.iloc[:window]])
        sm = ext.rolling(2 * window + 1, center=True).mean().iloc[window:-window]
        sm.columns = [f"clim_{c}" for c in sm.columns]
        sm["city"] = city
        sm.index.name = "doy"
        out.append(sm.reset_index())
    return pd.concat(out, ignore_index=True)


def add_targets(df: pd.DataFrame, clim: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    g = df.groupby("city", group_keys=False)
    for v in TARGETS:
        for h in HORIZONS:
            df[f"y_{v}_{h}"] = g[v].shift(-h)
    for h in HORIZONS:
        key = pd.DataFrame({"city": df["city"], "doy": (df["date"] + pd.Timedelta(days=h)).dt.dayofyear.clip(upper=365)})
        m = key.merge(clim, on=["city", "doy"], how="left")
        for v in TARGETS:
            df[f"clim_{v}_{h}"] = m[f"clim_{v}"].values
    return df


def feats_for(v, h):
    return BASE_FEATURES + [f"clim_{w}_{h}" for w in TARGETS]


def scores(y, yhat, tol):
    err = yhat - y
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "within_tol_pct": float(100 * np.mean(np.abs(err) <= tol)), "bias": float(np.mean(err))}


def main():
    out_dir = MODEL_DIR / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "daily_features.parquet")
    clim = weather_climatology(df)
    clim.to_parquet(MODEL_DIR / "weather_climatology.parquet", index=False)
    df = add_targets(df, clim)

    rows, preds = [], []
    for v, (label, unit, tol) in TARGETS.items():
        for h in HORIZONS:
            feats = feats_for(v, h)
            y = f"y_{v}_{h}"
            d = df.dropna(subset=[y, v])
            tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
            base = {"Persistence": te[v].values, "Climatology": te[f"clim_{v}_{h}"].values}
            for name, yhat in base.items():
                rows.append({"variable": v, "label": label, "unit": unit, "h": h, "model": name,
                             **scores(te[y].values, yhat, tol)})
            ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
            ridge.fit(tr[feats], tr[y])
            rows.append({"variable": v, "label": label, "unit": unit, "h": h, "model": "LinearRegression",
                         **scores(te[y].values, ridge.predict(te[feats]), tol)})
            m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=48, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  n_jobs=-1, random_state=SEED, verbose=-1)
            m.fit(tr[feats], tr[y], eval_set=[(va[feats], va[y])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            yhat = m.predict(te[feats])
            rows.append({"variable": v, "label": label, "unit": unit, "h": h, "model": "LightGBM",
                         **scores(te[y].values, yhat, tol)})
            joblib.dump({"model": m, "features": feats}, out_dir / f"{v}_h{h}.joblib", compress=3)
            preds.append(pd.DataFrame({"city": te["city"].values, "date": te["date"].values, "variable": v, "h": h,
                                       "y": te[y].values, "LightGBM": yhat,
                                       "Climatology": base["Climatology"], "Persistence": base["Persistence"]}))
            r = rows[-1]
            print(f"{label:16s} day {h}: LightGBM MAE {r['mae']:.2f} {unit}, within ±{tol:g} {unit}: {r['within_tol_pct']:.1f}%")

    met = pd.DataFrame(rows)
    met.to_csv(REPORT_DIR / "weather_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(REPORT_DIR / "weather_test_predictions.parquet", index=False)
    print(met.pivot_table(index=["label", "model"], columns="h", values="mae").round(2).to_string())


if __name__ == "__main__":
    main()
