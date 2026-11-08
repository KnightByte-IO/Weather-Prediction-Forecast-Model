"""Train and evaluate the daily (7-day) rain models.

For each horizon h = 1..7 days ahead we train:
  * classifiers  -> probability that day D+h gets >= 1 mm of rain
  * regressors   -> rainfall amount (mm) on day D+h
and compare them with two simple baselines (climatology and persistence).

Usage:  python src/train_daily.py
Output: models/daily/*.joblib, reports/daily_metrics.csv,
        reports/daily_test_predictions.parquet, reports/daily_feature_importance.csv
"""
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             mean_absolute_error, mean_squared_error,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from features import HORIZONS, features_for_horizon

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
SEED = 42


# ----------------------------------------------------------------- metrics
def classification_metrics(y, p, threshold, clim_p):
    yhat = (p >= threshold).astype(int)
    tp = ((yhat == 1) & (y == 1)).sum()
    fp = ((yhat == 1) & (y == 0)).sum()
    fn = ((yhat == 0) & (y == 1)).sum()
    brier = brier_score_loss(y, p)
    brier_clim = brier_score_loss(y, clim_p)
    return {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier,
        "brier_skill_vs_clim": 1 - brier / brier_clim,
        "accuracy": accuracy_score(y, yhat),
        "f1": f1_score(y, yhat, zero_division=0),
        "pod": tp / max(tp + fn, 1),          # probability of detection (recall)
        "far": fp / max(tp + fp, 1),          # false alarm ratio
        "csi": tp / max(tp + fp + fn, 1),     # critical success index
        "threshold": threshold,
    }


def best_threshold(y, p):
    grid = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def regression_metrics(y, yhat):
    wet = y >= 1.0
    return {
        "mae_mm": mean_absolute_error(y, yhat),
        "rmse_mm": mean_squared_error(y, yhat) ** 0.5,
        "mae_wet_days_mm": mean_absolute_error(y[wet], yhat[wet]) if wet.any() else np.nan,
        "bias_mm": float(np.mean(yhat - y)),
    }


# ------------------------------------------------------------------ models
def make_classifiers(n_pos, n_neg):
    return {
        "LogisticRegression": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5)),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                   n_jobs=-1, random_state=SEED)),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
            eval_metric="logloss", early_stopping_rounds=100, n_jobs=-1, random_state=SEED),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=3000, learning_rate=0.02, num_leaves=48, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=SEED, verbose=-1),
    }


def make_regressors():
    return {
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=20,
                                  n_jobs=-1, random_state=SEED)),
        "XGBoost": xgb.XGBRegressor(
            objective="reg:tweedie", tweedie_variance_power=1.4,
            n_estimators=2000, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, early_stopping_rounds=100, n_jobs=-1, random_state=SEED),
        "LightGBM": lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=1.4,
            n_estimators=3000, learning_rate=0.02, num_leaves=48, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            n_jobs=-1, random_state=SEED, verbose=-1),
    }


def fit(name, model, Xtr, ytr, Xva, yva):
    if name == "XGBoost":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif name == "LightGBM":
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    else:
        model.fit(Xtr, ytr)
    return model


# -------------------------------------------------------------------- main
def main():
    out_dir = MODEL_DIR / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DIR / "daily_features.parquet")

    rows, preds, importances = [], [], []
    for h in HORIZONS:
        feats = features_for_horizon(h)
        d = df.dropna(subset=[f"y_rain_{h}", "precip_sum"])
        tr, va, te = (d[d["split"] == s] for s in ("train", "val", "test"))
        Xtr, Xva, Xte = tr[feats], va[feats], te[feats]
        yc_tr, yc_va, yc_te = (x[f"y_rain_{h}"].astype(int) for x in (tr, va, te))
        ym_tr, ym_va, ym_te = (x[f"y_mm_{h}"].values for x in (tr, va, te))
        clim_va, clim_te = va[f"clim_p_{h}"].values, te[f"clim_p_{h}"].values
        print(f"\n=== horizon {h} day(s): train {len(tr):,}  val {len(va):,}  test {len(te):,}")

        pred_frame = te[["city", "date"]].copy()
        pred_frame["h"] = h
        pred_frame["y_rain"] = yc_te.values
        pred_frame["y_mm"] = ym_te

        # Baselines
        thr = best_threshold(yc_va, clim_va)
        rows.append({"h": h, "task": "rain_yes_no", "model": "Climatology",
                     **classification_metrics(yc_te.values, clim_te, thr, clim_te)})
        pers_va, pers_te = va["rainy"].values, te["rainy"].values
        rows.append({"h": h, "task": "rain_yes_no", "model": "Persistence",
                     **classification_metrics(yc_te.values, pers_te, 0.5, clim_te)})
        pred_frame["p_Climatology"] = clim_te
        rows.append({"h": h, "task": "rain_mm", "model": "Climatology",
                     **regression_metrics(ym_te, te[f"clim_mm_{h}"].values)})

        # Classifiers
        for name, model in make_classifiers(yc_tr.sum(), (1 - yc_tr).sum()).items():
            model = fit(name, model, Xtr, yc_tr, Xva, yc_va)
            p_va = model.predict_proba(Xva)[:, 1]
            p_te = model.predict_proba(Xte)[:, 1]
            thr = best_threshold(yc_va, p_va)
            m = classification_metrics(yc_te.values, p_te, thr, clim_te)
            rows.append({"h": h, "task": "rain_yes_no", "model": name, **m})
            pred_frame[f"p_{name}"] = p_te
            joblib.dump({"model": model, "features": feats, "threshold": thr},
                        out_dir / f"clf_{name}_h{h}.joblib", compress=3)
            print(f"  {name:18s} AUC {m['auc']:.3f}  Brier {m['brier']:.3f}  "
                  f"BSS {m['brier_skill_vs_clim']:+.3f}  F1 {m['f1']:.3f}")
            if name == "LightGBM":
                imp = pd.Series(model.booster_.feature_importance("gain"), index=feats)
                importances.append((imp / imp.sum()).rename(f"h{h}"))

        # Simple ensemble of the three tree models
        ens_cols = ["p_RandomForest", "p_XGBoost", "p_LightGBM"]
        p_te = pred_frame[ens_cols].mean(axis=1).values
        # threshold for the ensemble: re-predict on val
        p_va = np.mean([joblib.load(out_dir / f"clf_{n}_h{h}.joblib")["model"].predict_proba(Xva)[:, 1]
                        for n in ("RandomForest", "XGBoost", "LightGBM")], axis=0)
        thr = best_threshold(yc_va, p_va)
        rows.append({"h": h, "task": "rain_yes_no", "model": "Ensemble(RF+XGB+LGBM)",
                     **classification_metrics(yc_te.values, p_te, thr, clim_te)})
        pred_frame["p_Ensemble"] = p_te
        joblib.dump({"threshold": thr}, out_dir / f"ensemble_h{h}.joblib")

        # Regressors
        for name, model in make_regressors().items():
            model = fit(name, model, Xtr, ym_tr, Xva, ym_va)
            mm_te = np.clip(model.predict(Xte), 0, None)
            m = regression_metrics(ym_te, mm_te)
            rows.append({"h": h, "task": "rain_mm", "model": name, **m})
            pred_frame[f"mm_{name}"] = mm_te
            joblib.dump({"model": model, "features": feats}, out_dir / f"reg_{name}_h{h}.joblib", compress=3)
            print(f"  {name:18s} MAE {m['mae_mm']:.2f} mm  RMSE {m['rmse_mm']:.2f} mm")

        preds.append(pred_frame)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORT_DIR / "daily_metrics.csv", index=False)
    pd.concat(preds, ignore_index=True).to_parquet(REPORT_DIR / "daily_test_predictions.parquet", index=False)
    if importances:
        pd.concat(importances, axis=1).sort_values("h1", ascending=False).to_csv(
            REPORT_DIR / "daily_feature_importance.csv")
    pd.set_option("display.width", 200)
    print("\nTest-period AUC by horizon:")
    print(metrics[metrics.task == "rain_yes_no"].pivot(index="model", columns="h", values="auc").round(3))


if __name__ == "__main__":
    main()
