"""Build the daily modelling dataset from the raw hourly downloads.

Usage:  python src/build_dataset.py
Output: data/processed/daily_features.parquet
        data/processed/climatology.parquet
        reports/data_summary.csv
"""
import pandas as pd

from config import MODEL_DIR, PROCESSED_DIR, RAIN_THRESHOLD_MM, RAW_DIR, REPORT_DIR, TRAIN_CITIES
from features import (add_daily_features, add_targets, attach_climatology,
                      climatology_table, hourly_to_daily)

TRAIN_YEARS = range(2014, 2023)   # 2014-2022 train
VAL_YEARS = [2023]                # 2023 validation (tuning, early stopping)
# 2024-01-01 onwards is the untouched test period


def load_city(name: str) -> pd.DataFrame:
    return pd.read_parquet(RAW_DIR / f"{name.lower()}_hourly.parquet")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, meta in TRAIN_CITIES.items():
        path = RAW_DIR / f"{name.lower()}_hourly.parquet"
        if not path.exists():
            print(f"skip {name}: no raw file")
            continue
        h = load_city(name)
        d = hourly_to_daily(h)
        d = add_daily_features(d)
        d = add_targets(d)
        d["city"] = name
        d["lat"], d["lon"] = meta["lat"], meta["lon"]
        d["elevation"] = float(h["elevation"].iloc[0])
        d["region"] = meta["region"]
        frames.append(d)
        print(f"{name}: {len(d):,} days")

    daily = pd.concat(frames, ignore_index=True)
    clim = climatology_table(daily, years=TRAIN_YEARS)
    daily = attach_climatology(daily, clim)

    year = daily["date"].dt.year
    daily["split"] = "test"
    daily.loc[year.isin(TRAIN_YEARS), "split"] = "train"
    daily.loc[year.isin(VAL_YEARS), "split"] = "val"

    daily.to_parquet(PROCESSED_DIR / "daily_features.parquet", index=False)
    clim.to_parquet(PROCESSED_DIR / "climatology.parquet", index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clim.to_parquet(MODEL_DIR / "daily_climatology.parquet", index=False)  # used by the app

    # Short summary table for the report
    ok = daily.dropna(subset=["precip_sum"])
    ok = ok.assign(rainy=ok["precip_sum"] >= RAIN_THRESHOLD_MM,
                   monsoon=ok["date"].dt.month.isin([6, 7, 8, 9]))
    summ = ok.groupby("city").agg(
        region=("region", "first"),
        days=("date", "count"),
        first_day=("date", "min"),
        last_day=("date", "max"),
        mean_annual_rain_mm=("precip_sum", lambda s: s.sum() / (len(s) / 365.25)),
        rainy_day_pct=("rainy", lambda s: 100 * s.mean()),
    )
    mons = ok[ok["monsoon"]].groupby("city")["precip_sum"].sum()
    summ["monsoon_share_pct"] = 100 * mons / ok.groupby("city")["precip_sum"].sum()
    num = summ.select_dtypes("number").columns
    summ[num] = summ[num].round(1)
    summ = summ.sort_values("mean_annual_rain_mm", ascending=False)
    summ.to_csv(REPORT_DIR / "data_summary.csv")
    print(summ.to_string())
    print("\nRows per split:\n", daily["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
