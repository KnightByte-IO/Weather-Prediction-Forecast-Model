"""Train the 48-hour hourly rain model (LightGBM, lead time as a feature).

Usage:  python src/train_hourly.py
Output: models/hourly/*.joblib, reports/hourly_metrics.csv,
        reports/hourly_metrics_by_lead.csv
"""
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

from config import MODEL_DIR, RAW_DIR, REPORT_DIR, TRAIN_CITIES
from hourly_features import (HOURLY_FEATURES, MAX_LEAD, hourly_climatology,
                             make_samples, prepare_hourly)

TRAIN_YEARS = list(range(2014, 2023))
VAL_YEARS = [2023]
SEED = 42
warnings.filterwarnings("ignore", message=".*eval_set.*")


def load_all():
    frames = []
    for name, meta in TRAIN_CITIES.items():
        path = RAW_DIR / f"{name.lower()}_hourly.parquet"
        if not path.exists():
            continue
        h = pd.read_parquet(path)
        h["city"] = name
        hp = prepare_hourly(h)
        hp["city"] = name
        hp["lat"], hp["lon"] = meta["lat"], meta["lon"]
        hp["elevation"] = float(h["elevation"].iloc[0])
        frames.append(hp)
    return frames


def build(frames, clim, years, every_h, lead_fraction, rng):
    out = []
    for hp in frames:
        yr = hp["time"].dt.year
        mask = yr.isin(years) & (hp["time"].dt.hour % every_h == 0)
        s = make_samples(hp, mask.values, range(1, MAX_LEAD + 1), clim, rng, lead_fraction)
        s["lat"], s["lon"], s["elevation"] = hp["lat"].iloc[0], hp["lon"].iloc[0], hp["elevation"].iloc[0]
        out.append(s)
    s = pd.concat(out, ignore_index=True).dropna(subset=["y_wet"])
    s[HOURLY_FEATURES] = s[HOURLY_FEATURES].astype("float32")
    return s


def main():
    rng = np.random.default_rng(SEED)
    out_dir = MODEL_DIR / "hourly"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_all()
    clim = hourly_climatology(pd.concat(frames), TRAIN_YEARS)
    clim.to_parquet(out_dir / "hourly_climatology.parquet", index=False)

    test_years = sorted(set(pd.concat([f["time"] for f in frames]).dt.year) - set(TRAIN_YEARS) - set(VAL_YEARS))
    tr = build(frames, clim, TRAIN_YEARS, every_h=6, lead_fraction=0.5, rng=rng)
    va = build(frames, clim, VAL_YEARS, every_h=6, lead_fraction=0.5, rng=rng)
    te = build(frames, clim, test_years, every_h=3, lead_fraction=1.0, rng=rng)
    print(f"samples: train {len(tr):,}  val {len(va):,}  test {len(te):,}")

    X = HOURLY_FEATURES
    clf = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=127,
                             min_child_samples=200, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)
    clf.fit(tr[X], tr["y_wet"], eval_set=[(va[X], va["y_wet"])],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    reg = lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5,
                            n_estimators=3000, learning_rate=0.05, num_leaves=127,
                            min_child_samples=200, subsample=0.7, subsample_freq=1,
                            colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbose=-1)
    reg.fit(tr[X], tr["y_mm"], eval_set=[(va[X], va["y_mm"])],
            callbacks=[lgb.early_stopping(100, verbose=False)])
    joblib.dump({"model": clf, "features": X}, out_dir / "hourly_clf_LightGBM.joblib")
    joblib.dump({"model": reg, "features": X}, out_dir / "hourly_reg_LightGBM.joblib")

    te["p_LightGBM"] = clf.predict_proba(te[X])[:, 1]
    te["mm_LightGBM"] = np.clip(reg.predict(te[X]), 0, None)

    def summarise(g):
        return pd.Series({
            "n": len(g),
            "auc": roc_auc_score(g["y_wet"], g["p_LightGBM"]),
            "auc_climatology": roc_auc_score(g["y_wet"], g["clim_wet_p"]),
            "brier": brier_score_loss(g["y_wet"], g["p_LightGBM"]),
            "brier_climatology": brier_score_loss(g["y_wet"], g["clim_wet_p"]),
            "mae_mm": mean_absolute_error(g["y_mm"], g["mm_LightGBM"]),
        })

    by_lead = te.groupby("lead").apply(summarise, include_groups=False)
    by_lead["brier_skill_vs_clim"] = 1 - by_lead["brier"] / by_lead["brier_climatology"]
    by_lead.to_csv(REPORT_DIR / "hourly_metrics_by_lead.csv")
    te["block"] = pd.cut(te["lead"], [0, 6, 12, 24, 48], labels=["1-6h", "7-12h", "13-24h", "25-48h"])
    blocks = te.groupby("block", observed=True).apply(summarise, include_groups=False)
    blocks["brier_skill_vs_clim"] = 1 - blocks["brier"] / blocks["brier_climatology"]
    blocks.to_csv(REPORT_DIR / "hourly_metrics.csv")
    te[["city", "time", "lead", "target_time", "y_wet", "y_mm", "clim_wet_p",
        "p_LightGBM", "mm_LightGBM"]].to_parquet(REPORT_DIR / "hourly_test_predictions.parquet", index=False)
    print(blocks.round(3).to_string())


if __name__ == "__main__":
    main()
