"""
Download hourly weather history (2014 to Oct 2026) for 10 Indian cities
from the free Open-Meteo archive (ERA5 reanalysis).

How to run (needs Python 3.9+ and internet):
    pip install requests pandas
    python download_weather_data.py

It takes about 20-25 minutes (it waits between requests to stay inside
Open-Meteo's free limits). If it stops halfway, just run it again: it
resumes from where it stopped.

Result: a single file  rainfall_raw_data.zip  in the same folder.
Upload that zip file back to the Claude thread.
"""
import os
import sys
import time
import zipfile

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
START_YEAR, END_YEAR = 2014, 2026
END_DATE = "2026-10-31"
OUT_DIR = "weather_raw"
ZIP_NAME = "rainfall_raw_data.zip"

CITIES = {
    "Mumbai":    (19.0760, 72.8777),
    "Kochi":     (9.9312, 76.2673),
    "Chennai":   (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Nagpur":    (21.1458, 79.0882),
    "Delhi":     (28.6139, 77.2090),
    "Jaipur":    (26.9124, 75.7873),
    "Kolkata":   (22.5726, 88.3639),
    "Dhanbad":   (23.7957, 86.4304),
    "Guwahati":  (26.1445, 91.7362),
}

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "precipitation",
    "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "shortwave_radiation", "soil_moisture_0_to_7cm", "vapour_pressure_deficit",
]


def fetch(lat, lon, start, end):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(8):
        try:
            r = requests.get(URL, params=params, timeout=300)
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code == 429:
            reason = r.json().get("reason", "") if r.headers.get("content-type", "").startswith("application/json") else ""
            if "Daily" in reason:
                sys.exit("Open-Meteo daily limit reached for your internet connection. "
                         "Run the script again tomorrow (it will resume).")
            wait = 60 * (attempt + 1)
            print(f"    rate limited, waiting {wait}s ...", flush=True)
            time.sleep(wait)
            continue
        r.raise_for_status()
        js = r.json()
        df = pd.DataFrame(js["hourly"])
        df["elevation"] = js.get("elevation")
        return df
    raise RuntimeError("Too many failed attempts, please run the script again.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for city, (lat, lon) in CITIES.items():
        final = os.path.join(OUT_DIR, f"{city.lower()}_hourly.csv.gz")
        if os.path.exists(final):
            print(f"{city}: already done")
            continue
        parts = []
        for year in range(START_YEAR, END_YEAR + 1):
            part_file = os.path.join(OUT_DIR, f".{city.lower()}_{year}.csv.gz")
            if os.path.exists(part_file):
                parts.append(pd.read_csv(part_file))
                continue
            start = f"{year}-01-01"
            end = min(f"{year}-12-31", END_DATE)
            print(f"{city}: {year} ...", flush=True)
            df = fetch(lat, lon, start, end)
            df.to_csv(part_file, index=False)
            parts.append(df)
            time.sleep(8)
        full = pd.concat(parts, ignore_index=True)
        full.insert(0, "city", city)
        full.to_csv(final, index=False)
        for year in range(START_YEAR, END_YEAR + 1):
            p = os.path.join(OUT_DIR, f".{city.lower()}_{year}.csv.gz")
            if os.path.exists(p):
                os.remove(p)
        print(f"{city}: done, {len(full):,} hourly rows")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_STORED) as z:
        for city in CITIES:
            name = f"{city.lower()}_hourly.csv.gz"
            z.write(os.path.join(OUT_DIR, name), name)
    print(f"\nAll done. Upload {ZIP_NAME} ({os.path.getsize(ZIP_NAME)/1e6:.1f} MB) to the Claude thread.")


if __name__ == "__main__":
    main()
