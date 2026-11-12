#!/usr/bin/env bash
# Full pipeline: data -> features -> models -> report.
# Usage:  bash run_pipeline.sh [path/to/rainfall_raw_data.zip]
set -euo pipefail
cd "$(dirname "$0")"

if [ "${1:-}" != "" ]; then
  python src/import_raw_zip.py "$1"          # use a zip made by download_weather_data.py
elif ! ls data/raw/*_hourly.parquet >/dev/null 2>&1; then
  python src/download_data.py                # or download directly (needs Open-Meteo quota)
fi

python src/build_dataset.py      # hourly -> daily features + targets
python src/train_daily.py        # 7-day models: LogReg, Random Forest, XGBoost, LightGBM
python src/train_weather.py      # 7-day max/min temperature, humidity, wind (LightGBM)
python src/train_hourly.py       # 48-hour LightGBM model
python src/train_lstm.py both    # daily + hourly LSTM (PyTorch)
python src/evaluate.py           # figures + results_summary.md

echo "Done. Start the app with:  streamlit run app.py"
