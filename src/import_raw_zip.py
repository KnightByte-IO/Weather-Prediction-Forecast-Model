"""Convert the zip produced by download_weather_data.py into data/raw/*.parquet.

Usage:  python src/import_raw_zip.py path/to/rainfall_raw_data.zip
"""
import io
import sys
import zipfile

import pandas as pd

from config import RAW_DIR


def main(zip_path: str):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith("_hourly.csv.gz"):
                continue
            df = pd.read_csv(io.BytesIO(z.read(name)), compression="gzip")
            df["time"] = pd.to_datetime(df["time"])
            out = RAW_DIR / name.replace(".csv.gz", ".parquet")
            df.to_parquet(out, index=False)
            print(f"{name}: {len(df):,} rows, {df['time'].min()} to {df['time'].max()}")


if __name__ == "__main__":
    main(sys.argv[1])
