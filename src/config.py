"""Shared settings for the rainfall prediction project."""
import os
from pathlib import Path

# RAINFALL_ROOT lets tests point the pipeline at a scratch folder.
ROOT = Path(os.environ.get("RAINFALL_ROOT", Path(__file__).resolve().parents[1]))
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TIMEZONE = "Asia/Kolkata"
START_DATE = "2014-01-01"
END_DATE = "2026-10-31"

# A day counts as "rainy" at 1 mm or more, the IMD rainy-day threshold
# is 2.5 mm; 1 mm is the WMO "wet day" convention. We use 1.0 mm.
RAIN_THRESHOLD_MM = 1.0

# Training cities: chosen to cover India's main rainfall regimes.
TRAIN_CITIES = {
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "region": "West coast (SW monsoon)"},
    "Kochi":     {"lat": 9.9312,  "lon": 76.2673, "region": "Kerala (both monsoons)"},
    "Chennai":   {"lat": 13.0827, "lon": 80.2707, "region": "East coast (NE monsoon)"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "region": "Deccan plateau"},
    "Nagpur":    {"lat": 21.1458, "lon": 79.0882, "region": "Central India"},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090, "region": "North plains"},
    "Jaipur":    {"lat": 26.9124, "lon": 75.7873, "region": "Semi-arid north-west"},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639, "region": "Gangetic delta"},
    "Dhanbad":   {"lat": 23.7957, "lon": 86.4304, "region": "Chota Nagpur plateau"},
    "Guwahati":  {"lat": 26.1445, "lon": 91.7362, "region": "North-east (very wet)"},
}

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "shortwave_radiation",
    "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
