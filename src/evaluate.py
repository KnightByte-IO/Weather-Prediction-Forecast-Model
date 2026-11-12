"""Make report tables and figures from the saved test-period predictions.

Usage:  python src/evaluate.py
Output: reports/figures/*.png, reports/results_summary.md
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from config import PROCESSED_DIR, REPORT_DIR  # noqa: E402
from train_weather import TARGETS  # noqa: E402

FIG = REPORT_DIR / "figures"
# Fixed categorical order (validated palette): one colour per model, never re-used
COLORS = {
    "LSTM": "#2a78d6", "LightGBM": "#eb6834", "XGBoost": "#1baf7a",
    "RandomForest": "#eda100", "LogisticRegression": "#e87ba4", "Ensemble": "#6250d6",
    "Climatology": "#8a8985", "Persistence": "#c3c2b7",
}
plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "axes.titleweight": "bold", "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def daily_results():
    pred = pd.read_parquet(REPORT_DIR / "daily_test_predictions.parquet")
    lstm_path = REPORT_DIR / "lstm_daily_test_predictions.parquet"
    if lstm_path.exists():
        lstm = pd.read_parquet(lstm_path)
        pred = pred.merge(lstm, on=["city", "date", "h"], how="left")
    model_cols = [c for c in pred if c.startswith("p_")]
    rows = []
    for h, g in pred.groupby("h"):
        g = g.dropna(subset=model_cols)
        clim_brier = brier_score_loss(g["y_rain"], g["p_Climatology"])
        if "p_LSTM" in g:
            g = g.assign(p_FinalEnsemble=g[["p_RandomForest", "p_XGBoost", "p_LightGBM", "p_LSTM"]].mean(axis=1))
        for c in [c for c in g if c.startswith("p_")]:
            b = brier_score_loss(g["y_rain"], g[c])
            rows.append({"h": h, "model": c[2:], "auc": roc_auc_score(g["y_rain"], g[c]),
                         "brier": b, "bss": 1 - b / clim_brier, "n": len(g)})
    return pred, pd.DataFrame(rows)


def plot_by_horizon(res, metric, ylabel, fname, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, g in res[res.model != "Ensemble"].groupby("model"):  # tree-only ensemble: table only
        color = COLORS.get(model.replace("Final", ""), "#52514e")
        style = "--" if model in ("Climatology", "Persistence") else "-"
        lw = 3 if model == "FinalEnsemble" else 2
        ax.plot(g["h"], g[metric], style, marker="o", color=color, lw=lw, label=model)
    ax.set_xlabel("Days ahead (1 = today)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(range(1, 8))
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def reliability(pred, model="p_LightGBM", h=1):
    g = pred[pred["h"] == h].dropna(subset=[model])
    bins = np.linspace(0, 1, 11)
    g = g.assign(bin=pd.cut(g[model], bins, include_lowest=True))
    t = g.groupby("bin", observed=True).agg(p=(model, "mean"), o=("y_rain", "mean"), n=("y_rain", "size"))
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="#8a8985", lw=1, label="Perfect")
    ax.plot(t["p"], t["o"], marker="o", color=COLORS["LightGBM"], label=model[2:])
    ax.set_xlabel("Predicted chance of rain")
    ax.set_ylabel("Observed frequency of rain")
    ax.set_title(f"Reliability, day {h}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"reliability_day{h}.png")
    plt.close(fig)


def feature_importance():
    p = REPORT_DIR / "daily_feature_importance.csv"
    if not p.exists():
        return
    imp = pd.read_csv(p, index_col=0)["h1"].sort_values().tail(20)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(imp.index, imp.values * 100, color=COLORS["LightGBM"], height=0.6)
    ax.set_xlabel("Share of total gain (%)")
    ax.set_title("Top 20 features, LightGBM, day 1")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "feature_importance_day1.png")
    plt.close(fig)


def monthly_climate():
    d = pd.read_parquet(PROCESSED_DIR / "daily_features.parquet", columns=["city", "date", "precip_sum"])
    d = d.dropna()
    years = d.groupby("city")["date"].apply(lambda s: s.dt.year.nunique())
    m = d.assign(month=d["date"].dt.month).groupby(["city", "month"])["precip_sum"].sum().unstack()
    m = m.div(years, axis=0)
    cities = m.sum(axis=1).sort_values(ascending=False).index
    n = len(cities)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows), sharey=True)
    for ax, c in zip(np.ravel(axes), cities):
        ax.bar(range(1, 13), m.loc[c], color="#2a78d6", width=0.7)
        ax.set_title(c, fontsize=10)
        ax.set_xticks([1, 4, 7, 10])
        ax.set_xticklabels(["Jan", "Apr", "Jul", "Oct"])
        ax.grid(axis="x", visible=False)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.supylabel("Average rain (mm / month)")
    fig.suptitle("Monthly rainfall by city, 2014-2026 (ERA5)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "monthly_rainfall_by_city.png")
    plt.close(fig)


def example_timeseries(pred, city="Mumbai", year=2024):
    g = pred[(pred["city"] == city) & (pred["h"] == 1)].copy()
    g["target"] = g["date"] + pd.Timedelta(days=1)
    g = g[(g["target"].dt.year == year) & g["target"].dt.month.isin([6, 7, 8, 9])]
    if g.empty:
        return
    col = "p_FinalEnsemble" if "p_FinalEnsemble" in g else "p_Ensemble"
    if col not in g:
        g[col] = g[["p_RandomForest", "p_XGBoost", "p_LightGBM"] + (["p_LSTM"] if "p_LSTM" in g else [])].mean(axis=1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axes[0].bar(g["target"], g["y_mm"], color="#2a78d6", width=0.8, label="Actual")
    axes[0].set_ylabel("Actual rain (mm)")
    axes[0].set_title(f"{city}, monsoon {year}: actual rain vs ML chance of rain (1 day ahead)")
    axes[1].plot(g["target"], 100 * g[col], color="#eb6834")
    axes[1].set_ylabel("ML chance of rain (%)")
    axes[1].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG / f"example_{city.lower()}_{year}.png")
    plt.close(fig)


def hourly_plot():
    p = REPORT_DIR / "hourly_metrics_by_lead.csv"
    if not p.exists():
        return
    m = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(m["lead"], m["auc"], color=COLORS["LightGBM"], label="LightGBM")
    ax.plot(m["lead"], m["auc_climatology"], "--", color=COLORS["Climatology"], label="Climatology")
    ax.set_xlabel("Hours ahead")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Hourly rain model: skill by lead time")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "hourly_auc_by_lead.png")
    plt.close(fig)


def rain_accuracy_plot():
    """Share of days with the rain / no-rain call right, by lead day."""
    m = pd.read_csv(REPORT_DIR / "daily_metrics.csv")
    m = m[m.task == "rain_yes_no"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, color, style in (("Ensemble(RF+XGB+LGBM)", COLORS["Ensemble"], "-"),
                                ("Persistence", COLORS["Persistence"], "--"),
                                ("Climatology", COLORS["Climatology"], "--")):
        g = m[m.model == model].sort_values("h")
        ax.plot(g["h"], 100 * g["accuracy"], style, marker="o", color=color,
                label="ML ensemble" if model.startswith("Ensemble") else model)
    ax.set_xlabel("Days ahead (1 = today)")
    ax.set_ylabel("Days called correctly (%)")
    ax.set_title("Rain / no rain: accuracy on 2024-2026 test data")
    ax.set_xticks(range(1, 8))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "rain_accuracy_by_day.png")
    plt.close(fig)


def weather_plot():
    """Temperature, humidity and wind: mean absolute error by lead day, one panel per variable."""
    p = REPORT_DIR / "weather_metrics.csv"
    if not p.exists():
        return None
    m = pd.read_csv(p)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, (var, g) in zip(np.ravel(axes), m.groupby("variable", sort=False)):
        for model, color, style in (("LightGBM", COLORS["LightGBM"], "-"),
                                    ("LinearRegression", COLORS["LogisticRegression"], "-"),
                                    ("Persistence", COLORS["Persistence"], "--"),
                                    ("Climatology", COLORS["Climatology"], "--")):
            x = g[g.model == model].sort_values("h")
            ax.plot(x["h"], x["mae"], style, marker="o", color=color, label=model)
        ax.set_title(f"{g.label.iloc[0]} ({g.unit.iloc[0]})")
        ax.set_xticks(range(1, 8))
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Days ahead (1 = today)")
    fig.supylabel("Mean absolute error (lower is better)")
    fig.suptitle("Temperature, humidity and wind forecasts on 2024-2026 test data", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "weather_mae_by_day.png")
    plt.close(fig)
    return m


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    pred, res = daily_results()
    res.to_csv(REPORT_DIR / "daily_results_with_lstm.csv", index=False)
    plot_by_horizon(res, "auc", "ROC-AUC (higher is better)", "daily_auc_by_horizon.png",
                    "Rain / no rain: ROC-AUC on 2024-2026 test data")
    plot_by_horizon(res[~res.model.isin(["Persistence"])], "bss", "Brier skill vs climatology",
                    "daily_brier_skill_by_horizon.png", "Improvement over climatology")
    pred["p_FinalEnsemble"] = pred[[c for c in ("p_RandomForest", "p_XGBoost", "p_LightGBM", "p_LSTM")
                                    if c in pred]].mean(axis=1)
    reliability(pred, "p_LightGBM", 1)
    feature_importance()
    monthly_climate()
    example_timeseries(pred, "Mumbai", 2024)
    hourly_plot()
    rain_accuracy_plot()
    wmet = weather_plot()

    auc = res.pivot(index="model", columns="h", values="auc").round(3)
    bss = res.pivot(index="model", columns="h", values="bss").round(3)
    auc.columns = bss.columns = [f"Day {h}" for h in auc.columns]
    lines = ["# Results summary (test period: Jan 2024 - Oct 2026)", "",
             "## Rain / no rain, ROC-AUC", "", auc.to_markdown(), "",
             "## Brier skill score vs climatology (above 0 = better than climatology)", "", bss.to_markdown(), ""]
    reg = pd.read_csv(REPORT_DIR / "daily_metrics.csv")
    reg = reg[reg.task == "rain_mm"].pivot(index="model", columns="h", values="mae_mm").round(2)
    reg.columns = [f"Day {h}" for h in reg.columns]
    lines += ["## Rain amount, mean absolute error (mm)", "", reg.to_markdown(), ""]
    acc = pd.read_csv(REPORT_DIR / "daily_metrics.csv")
    acc = (100 * acc[acc.task == "rain_yes_no"].pivot(index="model", columns="h", values="accuracy")).round(1)
    acc.columns = [f"Day {h}" for h in acc.columns]
    lines += ["## Rain / no rain, accuracy (% of days called correctly)", "", acc.to_markdown(), ""]
    if wmet is not None:
        for col, name in (("mae", "mean absolute error"), ("within_tol_pct", "% of days within tolerance")):
            for var, g in wmet.groupby("variable", sort=False):
                t = g.pivot(index="model", columns="h", values=col).round(2 if col == "mae" else 1)
                t.columns = [f"Day {h}" for h in t.columns]
                unit = g.unit.iloc[0]
                tol = f" (within ±{TARGETS[var][2]:g} {unit})" if col != "mae" else f" ({unit})"
                lines += [f"## {g.label.iloc[0]}, {name}{tol}", "", t.to_markdown(), ""]
    hp = REPORT_DIR / "hourly_metrics.csv"
    if hp.exists():
        lines += ["## Hourly model (next 48 h)", "", pd.read_csv(hp).round(3).to_markdown(index=False), ""]
    lp = REPORT_DIR / "lstm_hourly_metrics.csv"
    if lp.exists():
        lines += ["## Hourly LSTM", "", pd.read_csv(lp).round(3).to_markdown(index=False), ""]
    (REPORT_DIR / "results_summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
