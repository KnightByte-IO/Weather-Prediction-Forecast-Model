"""Download hourly historical weather (ERA5 reanalysis via Open-Meteo) for the training cities.

Usage:  python src/download_data.py
Output: data/raw/<city>_hourly.parquet  (one file per city)
"""
import sys
import time

import pandas as pd
import requests

from config import (ARCHIVE_URL, END_DATE, HOURLY_VARS, RAW_DIR, START_DATE,
                    TIMEZONE, TRAIN_CITIES)


def fetch_city(name: str, lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
    }
    for attempt in range(6):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=300)
        except (requests.ConnectionError, requests.Timeout) as exc:
            wait = 2 ** (attempt + 2)
            print(f"  network error ({type(exc).__name__}), retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code == 429:  # rate limited: wait and retry
            wait = 90 * (attempt + 1)
            print(f"  rate limited ({r.text[:120]}), waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        js = r.json()
        df = pd.DataFrame(js["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df.insert(0, "city", name)
        df.attrs["elevation"] = js.get("elevation")
        df["elevation"] = js.get("elevation")
        return df
    raise RuntimeError(f"Gave up on {name} {start}..{end} after repeated failures")


def main(cities=None):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cities = cities or list(TRAIN_CITIES)
    for i, name in enumerate(cities):
        out = RAW_DIR / f"{name.lower()}_hourly.parquet"
        if out.exists():
            print(f"{name}: already downloaded")
            continue
        c = TRAIN_CITIES[name]
        print(f"{name}: downloading {START_DATE} to {END_DATE} ...", flush=True)
        # One request per year keeps each call small enough for the free API limits.
        parts = []
        y0, y1 = int(START_DATE[:4]), int(END_DATE[:4])
        for y in range(y0, y1 + 1):
            s = max(START_DATE, f"{y}-01-01")
            e = min(END_DATE, f"{y}-12-31")
            parts.append(fetch_city(name, c["lat"], c["lon"], s, e))
            time.sleep(8)
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(out, index=False)
        print(f"{name}: {len(df):,} hourly rows saved", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
