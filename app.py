"""Streamlit dashboard: ML weather forecast (rain, temperature, humidity, wind) for Indian places.

Run:  streamlit run app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cities import INDIA_CITIES  # noqa: E402
from config import REPORT_DIR, TRAIN_CITIES  # noqa: E402
from predict import forecast_for  # noqa: E402

st.set_page_config(page_title="Weather Forecast India (ML)", page_icon="🌦️", layout="wide")

ML_COLOR, NWP_COLOR = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
LAYOUT = dict(height=340, margin=dict(t=85, b=10, l=10, r=10), hovermode="x unified",
              legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0),
              xaxis=dict(showgrid=False), plot_bgcolor="rgba(0,0,0,0)")


def chance_chart(x, ml_prob, official_pct, title, bars=False):
    """Chance of rain (%): ML vs official, one axis."""
    fig = go.Figure()
    if bars:
        fig.add_bar(x=x, y=100 * ml_prob, name="ML models", marker_color=ML_COLOR,
                    marker_cornerradius=4, hovertemplate="%{y:.0f}%")
        if official_pct is not None:
            fig.add_bar(x=x, y=official_pct, name="Official forecast", marker_color=NWP_COLOR,
                        marker_cornerradius=4, hovertemplate="%{y:.0f}%")
    else:
        fig.add_scatter(x=x, y=100 * ml_prob, name="ML models", line=dict(color=ML_COLOR, width=2),
                        hovertemplate="%{y:.0f}%")
        if official_pct is not None:
            fig.add_scatter(x=x, y=official_pct, name="Official forecast",
                            line=dict(color=NWP_COLOR, width=2), hovertemplate="%{y:.0f}%")
    fig.update_layout(title=dict(text=title, y=0.97, yanchor="top"), barmode="group", bargap=0.3, bargroupgap=0.1,
                      yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor="rgba(128,128,128,0.15)"),
                      **LAYOUT)
    return fig


def amount_chart(x, ml_mm, official_mm, title, unit="mm", digits=1):
    """One quantity (rain mm, humidity %, wind km/h): ML vs official as grouped bars, one axis."""
    fig = go.Figure()
    hover = f"%{{y:.{digits}f}} {unit}"
    fig.add_bar(x=x, y=ml_mm, name="ML models", marker_color=ML_COLOR, marker_cornerradius=4,
                hovertemplate=hover)
    if official_mm is not None:
        fig.add_bar(x=x, y=official_mm, name="Official forecast", marker_color=NWP_COLOR,
                    marker_cornerradius=4, hovertemplate=hover)
    fig.update_layout(title=dict(text=title, y=0.97, yanchor="top"), barmode="group", bargap=0.3, bargroupgap=0.1,
                      yaxis=dict(rangemode="tozero", ticksuffix=f" {unit}", gridcolor="rgba(128,128,128,0.15)"),
                      **LAYOUT)
    return fig


def temperature_chart(x, ml_max, ml_min, off_max, off_min, title):
    """Daily max and min temperature (°C): colour = source, solid = max, dotted = min."""
    fig = go.Figure()
    fig.add_scatter(x=x, y=ml_max, name="ML max", line=dict(color=ML_COLOR, width=2),
                    mode="lines+markers", hovertemplate="%{y:.1f} °C")
    fig.add_scatter(x=x, y=ml_min, name="ML min", line=dict(color=ML_COLOR, width=2, dash="dot"),
                    mode="lines+markers", hovertemplate="%{y:.1f} °C")
    if off_max is not None:
        fig.add_scatter(x=x, y=off_max, name="Official max", line=dict(color=NWP_COLOR, width=2),
                        mode="lines+markers", hovertemplate="%{y:.1f} °C")
        fig.add_scatter(x=x, y=off_min, name="Official min", line=dict(color=NWP_COLOR, width=2, dash="dot"),
                        mode="lines+markers", hovertemplate="%{y:.1f} °C")
    fig.update_layout(title=dict(text=title, y=0.97, yanchor="top"),
                      yaxis=dict(ticksuffix=" °C", gridcolor="rgba(128,128,128,0.15)"), **LAYOUT)
    return fig



@st.cache_data(ttl=1800, show_spinner=False)
def get_forecast(lat, lon):
    return forecast_for(lat, lon)


def rain_label(p):
    if p >= 0.7:
        return "🌧️ Rain likely"
    if p >= 0.4:
        return "🌦️ Possible showers"
    if p >= 0.2:
        return "⛅ Low chance"
    return "☀️ Mostly dry"


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🌦️ Weather Forecast")
mode = st.sidebar.radio("Location", ["Choose a city", "Enter coordinates"])
if mode == "Choose a city":
    names = sorted(INDIA_CITIES)
    city = st.sidebar.selectbox("City", names, index=names.index("Dhanbad"))
    lat, lon = INDIA_CITIES[city]
else:
    city = "Custom location"
    lat = st.sidebar.number_input("Latitude", 6.0, 37.5, 23.80, 0.01)
    lon = st.sidebar.number_input("Longitude", 68.0, 97.5, 86.43, 0.01)
show_official = st.sidebar.checkbox("Compare with official weather forecast", value=True)
st.sidebar.caption("ML models: Random Forest, XGBoost, LightGBM and LSTM for rain, LightGBM for "
                   "temperature, humidity and wind, all trained on 2014-2022 ERA5 data for 10 Indian "
                   "cities. Official forecast: Open-Meteo.")

# ------------------------------------------------------------------- data
try:
    with st.spinner("Fetching latest weather and running ML models..."):
        fc = get_forecast(lat, lon)
except Exception as exc:  # network errors, missing models
    st.error(f"Could not make a forecast: {exc}")
    st.stop()

daily, hourly = fc["daily"], fc["hourly"]
has_weather = "ml_t_max" in daily
st.title(f"{city} weather forecast")
st.caption(f"Latest data: {fc['data_until']:%d %b %Y, %H:%M} IST · elevation {fc['elevation']:.0f} m · "
           f"local climate profile from {fc['climatology_city']} ({fc['climatology_km']:.0f} km away)")

# ------------------------------------------------------- today & tomorrow
today, tomorrow = daily.iloc[0], daily.iloc[1]
c1, c2 = st.columns(2)
for col, row, label in ((c1, today, "Today"), (c2, tomorrow, "Tomorrow")):
    with col:
        st.subheader(f"{label} · {row['date']:%a %d %b}")
        a, b = st.columns(2)
        a.metric("ML chance of rain", f"{100 * row['ml_prob']:.0f}%")
        b.metric("ML expected rain", f"{row['ml_mm']:.1f} mm")
        if has_weather:
            a, b, c = st.columns(3)
            a.metric("ML max / min", f"{row['ml_t_max']:.0f}° / {row['ml_t_min']:.0f}°C")
            b.metric("ML humidity", f"{row['ml_rh_mean']:.0f}%")
            c.metric("ML max wind", f"{row['ml_ws_max']:.0f} km/h")
        st.markdown(f"**{rain_label(row['ml_prob'])}**")
        if show_official and pd.notna(row.get("precipitation_probability_max")):
            extra = ""
            if pd.notna(row.get("relative_humidity_2m_mean")):
                extra = (f" · humidity {row['relative_humidity_2m_mean']:.0f}%"
                         f" · wind up to {row['wind_speed_10m_max']:.0f} km/h")
            st.caption(f"Official forecast: {row['precipitation_probability_max']:.0f}% chance, "
                       f"{row['precipitation_sum']:.1f} mm · "
                       f"{row['temperature_2m_min']:.0f}–{row['temperature_2m_max']:.0f} °C{extra}")

# ------------------------------------------------------------- 7-day view
st.header("7-day outlook")
cols = st.columns(len(daily))
for col, (_, r) in zip(cols, daily.iterrows()):
    with col:
        st.markdown(f"**{r['date']:%a}**  \n{r['date']:%d %b}")
        st.markdown(f"### {100 * r['ml_prob']:.0f}%")
        st.caption(f"{r['ml_mm']:.1f} mm  \n{rain_label(r['ml_prob'])}")
        if has_weather:
            st.caption(f"{r['ml_t_min']:.0f}–{r['ml_t_max']:.0f} °C  \n💧 {r['ml_rh_mean']:.0f}%  \n"
                       f"💨 {r['ml_ws_max']:.0f} km/h")
        elif pd.notna(r.get("temperature_2m_max")):
            st.caption(f"{r['temperature_2m_min']:.0f}–{r['temperature_2m_max']:.0f} °C (official)")

left, right = st.columns(2)
with left:
    st.plotly_chart(chance_chart(daily["date"], daily["ml_prob"],
                                 daily["precipitation_probability_max"] if show_official else None,
                                 "Chance of rain by day", bars=True), width="stretch")
with right:
    st.plotly_chart(amount_chart(daily["date"], daily["ml_mm"],
                                 daily["precipitation_sum"] if show_official else None,
                                 "Expected rain by day (mm)"), width="stretch")

if has_weather:
    st.plotly_chart(temperature_chart(daily["date"], daily["ml_t_max"], daily["ml_t_min"],
                                      daily["temperature_2m_max"] if show_official else None,
                                      daily["temperature_2m_min"] if show_official else None,
                                      "Max and min temperature by day"), width="stretch")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(amount_chart(daily["date"], daily["ml_rh_mean"],
                                     daily.get("relative_humidity_2m_mean") if show_official else None,
                                     "Mean humidity by day (%)", unit="%", digits=0), width="stretch")
    with right:
        st.plotly_chart(amount_chart(daily["date"], daily["ml_ws_max"],
                                     daily.get("wind_speed_10m_max") if show_official else None,
                                     "Max wind speed by day (km/h)", unit="km/h", digits=0),
                        width="stretch")

with st.expander("What each ML model predicts (7 days)"):
    show = daily[["date"] + [c for c in daily if c.startswith("p_")]].copy()
    show["date"] = show["date"].dt.strftime("%a %d %b")
    for c in show.columns[1:]:
        show[c] = (100 * show[c]).round(0).astype(int).astype(str) + "%"
    show.columns = [c.replace("p_", "") for c in show.columns]
    st.dataframe(show.set_index("date"), width="stretch")

# --------------------------------------------------------- hourly detail
st.header("Hour by hour: next 48 hours")
st.plotly_chart(chance_chart(hourly["time"], hourly["ml_prob"],
                             hourly["precipitation_probability"] if show_official else None,
                             "Chance of rain by hour"), width="stretch")
st.plotly_chart(amount_chart(hourly["time"], hourly["ml_mm"],
                             hourly["precipitation"] if show_official else None,
                             "Expected rain by hour (mm)"), width="stretch")

table = pd.DataFrame({
    "Time": hourly["time"].dt.strftime("%a %H:%M"),
    "ML chance of rain": (100 * hourly["ml_prob"]).round(0).astype(int).astype(str) + "%",
    "ML rain (mm)": hourly["ml_mm"].round(1),
    "Temp (°C)": hourly["temperature_2m"].round(1),
    "Humidity (%)": hourly["relative_humidity_2m"].round(0),
    "Cloud (%)": hourly["cloud_cover"].round(0),
    "Wind (km/h)": hourly["wind_speed_10m"].round(0),
})
if show_official:
    table["Official chance"] = hourly["precipitation_probability"].astype("Int64").astype(str) + "%"
st.caption("Temperature, humidity, cloud and wind come from the official forecast; rain columns marked ML "
           "come from our models.")
st.dataframe(table.set_index("Time"), width="stretch", height=420)

# ------------------------------------------------------------ model card
with st.expander("How accurate are the models?"):
    p = REPORT_DIR / "daily_metrics.csv"
    if p.exists():
        m = pd.read_csv(p)
        auc = m[m.task == "rain_yes_no"].pivot(index="model", columns="h", values="auc").round(3)
        auc.columns = [f"Day {h}" for h in auc.columns]
        st.markdown("**ROC-AUC on the 2024–2026 test period** (1.0 = perfect, 0.5 = guessing)")
        st.dataframe(auc, width="stretch")
    p = REPORT_DIR / "weather_metrics.csv"
    if p.exists():
        w = pd.read_csv(p)
        w = w[w.model == "LightGBM"]
        mae = w.pivot(index="label", columns="h", values="mae").round(2)
        mae.columns = [f"Day {h}" for h in mae.columns]
        mae.index = [f"{i} ({w[w.label == i].unit.iloc[0]})" for i in mae.index]
        st.markdown("**Temperature, humidity and wind: average error of the ML forecast** (test period)")
        st.dataframe(mae, width="stretch")
    st.markdown(f"Training cities: {', '.join(TRAIN_CITIES)}.")
