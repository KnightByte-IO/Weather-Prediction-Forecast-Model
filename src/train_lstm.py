"""Deep-learning models (PyTorch LSTM) for the daily 7-day forecast and the 48-hour forecast.

Daily LSTM : last 30 days of daily weather  -> rain probability + mm for each of the next 7 days
Hourly LSTM: last 72 hours of hourly weather -> rain probability + mm for each of the next 48 hours

Usage:  python src/train_lstm.py [daily|hourly|both]
Output: models/lstm/*.pt, reports/lstm_daily_metrics.csv, reports/lstm_hourly_metrics.csv,
        reports/lstm_daily_test_predictions.parquet
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

from config import MODEL_DIR, PROCESSED_DIR, RAIN_THRESHOLD_MM, RAW_DIR, REPORT_DIR, TRAIN_CITIES
from features import HORIZONS
from hourly_features import MAX_LEAD, WET_HOUR_MM, prepare_hourly

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = MODEL_DIR / "lstm"

DAILY_SEQ_COLS = [
    "precip_sum", "precip_hours", "precip_max_1h", "t_max", "t_min", "t_mean",
    "rh_mean", "rh_max", "rh_min", "td_mean", "dew_dep", "p_msl_mean", "p_msl_min",
    "p_msl_max", "sp_mean", "cc_mean", "cc_low_mean", "ws_mean", "gust_max",
    "u_mean", "v_mean", "rad_sum", "sm_mean", "vpd_mean", "doy_sin", "doy_cos",
]
HOURLY_SEQ_COLS = [
    "precipitation", "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "pressure_msl", "cloud_cover", "cloud_cover_low", "wind_speed_10m", "u", "v",
    "wind_gusts_10m", "shortwave_radiation", "soil_moisture_0_to_7cm",
    "vapour_pressure_deficit",
]
STATIC_COLS = ["lat", "lon", "elevation"]


class SeqModel(nn.Module):
    """LSTM encoder + static features -> two heads (rain probability logits, log1p mm)."""

    def __init__(self, n_in, n_static, n_out, hidden=96, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 128), nn.ReLU(), nn.Dropout(dropout),
        )
        self.logit = nn.Linear(128, n_out)
        self.mm = nn.Linear(128, n_out)

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        z = self.head(torch.cat([h[-1], static], dim=1))
        return self.logit(z), self.mm(z)


class WindowDataset(torch.utils.data.Dataset):
    """Windows cut on the fly from per-city arrays (no huge pre-built tensor)."""

    def __init__(self, arrays, statics, targets_p, targets_mm, index, seq_len, n_out):
        self.arrays, self.statics = arrays, statics
        self.tp, self.tm = targets_p, targets_mm
        self.index, self.seq_len, self.n_out = index, seq_len, n_out

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        c, t = self.index[i]  # t = last observed position
        seq = self.arrays[c][t - self.seq_len + 1: t + 1]
        yp = self.tp[c][t + 1: t + 1 + self.n_out]
        ym = self.tm[c][t + 1: t + 1 + self.n_out]
        return (torch.from_numpy(seq), torch.from_numpy(self.statics[c]),
                torch.from_numpy(yp), torch.from_numpy(ym))


def valid_index(arrays, targets, times, years, seq_len, n_out, stride, hour_mod=None):
    idx = []
    for c, (a, y, tm) in enumerate(zip(arrays, targets, times)):
        ok_x = ~np.isnan(a).any(axis=1)
        ok_y = ~np.isnan(y)
        # cumulative sums let us check whole windows quickly
        cx = np.concatenate([[0], np.cumsum(~ok_x)])
        cy = np.concatenate([[0], np.cumsum(~ok_y)])
        yrs = pd.DatetimeIndex(tm).year
        for t in range(seq_len - 1, len(a) - n_out, stride):
            if yrs[t] not in years:
                continue
            if hour_mod is not None and pd.Timestamp(tm[t]).hour % hour_mod != hour_mod - 1:
                continue
            # allow up to 10% missing input steps (filled with the mean), but the last
            # observed step and every target must be present
            if (ok_x[t] and cx[t + 1] - cx[t - seq_len + 1] <= seq_len // 10
                    and cy[t + 1 + n_out] - cy[t + 1] == 0):
                idx.append((c, t))
    return idx


def run_training(model, train_ds, val_ds, epochs, lr, name, pos_weight=None):
    tl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    epochs = int(os.environ.get("MAX_EPOCHS", epochs))
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for seq, st, yp, ym in tl:
            seq, st, yp, ym = seq.to(DEVICE), st.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
            lo, mm = model(seq, st)
            loss = bce(lo, yp) + 0.5 * mse(mm, torch.log1p(ym))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(seq)
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for seq, st, yp, ym in vl:
                seq, st, yp, ym = seq.to(DEVICE), st.to(DEVICE), yp.to(DEVICE), ym.to(DEVICE)
                lo, mm = model(seq, st)
                vtot += (bce(lo, yp) + 0.5 * mse(mm, torch.log1p(ym))).item() * len(seq)
        vloss = vtot / len(val_ds)
        sched.step(vloss)
        print(f"  [{name}] epoch {ep + 1:2d}  train {tot / len(train_ds):.4f}  val {vloss:.4f}", flush=True)
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 4:
                break
    model.load_state_dict(best_state)
    return model


def predict(model, ds):
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False)
    ps, ms, yps, yms = [], [], [], []
    model.eval()
    with torch.no_grad():
        for seq, st, yp, ym in dl:
            lo, mm = model(seq.to(DEVICE), st.to(DEVICE))
            ps.append(torch.sigmoid(lo).cpu().numpy())
            ms.append(np.expm1(mm.cpu().numpy().clip(0, 6.2)))
            yps.append(yp.numpy())
            yms.append(ym.numpy())
    return np.vstack(ps), np.vstack(ms), np.vstack(yps), np.vstack(yms)


def scaler_fit(arrays, idx, seq_len):
    rows = np.vstack([arrays[c][t - seq_len + 1: t + 1] for c, t in idx[:: max(1, len(idx) // 3000)]])
    mu, sd = np.nanmean(rows, 0), np.nanstd(rows, 0) + 1e-6
    return mu.astype("float32"), sd.astype("float32")


# ------------------------------------------------------------------- daily
def train_daily():
    df = pd.read_parquet(PROCESSED_DIR / "daily_features.parquet")
    arrays, statics, tp, tm, times, cities = [], [], [], [], [], []
    for city, g in df.groupby("city"):
        g = g.sort_values("date")
        arrays.append(g[DAILY_SEQ_COLS].to_numpy("float32"))
        statics.append(g[STATIC_COLS].iloc[0].to_numpy("float32"))
        mm = g["precip_sum"].to_numpy("float32")
        tm.append(mm)
        tp.append(np.where(np.isnan(mm), np.nan, (mm >= RAIN_THRESHOLD_MM)).astype("float32"))
        times.append(g["date"].to_numpy())
        cities.append(city)
    seq_len, n_out = 30, len(HORIZONS)
    years_tr, years_va = set(range(2014, 2023)), {2023}
    years_te = set(pd.DatetimeIndex(np.concatenate(times)).year) - years_tr - years_va
    itr = valid_index(arrays, tp, times, years_tr, seq_len, n_out, stride=1)
    iva = valid_index(arrays, tp, times, years_va, seq_len, n_out, stride=1)
    ite = valid_index(arrays, tp, times, years_te, seq_len, n_out, stride=1)
    mu, sd = scaler_fit(arrays, itr, seq_len)
    arrays_s = [np.nan_to_num((a - mu) / sd).astype("float32") for a in arrays]
    smu = np.mean(statics, 0)
    ssd = np.std(statics, 0) + 1e-6
    statics_s = [((s - smu) / ssd).astype("float32") for s in statics]
    mk = lambda ix: WindowDataset(arrays_s, statics_s, tp, tm, ix, seq_len, n_out)
    print(f"daily LSTM windows: train {len(itr):,} val {len(iva):,} test {len(ite):,}")

    model = SeqModel(len(DAILY_SEQ_COLS), len(STATIC_COLS), n_out).to(DEVICE)
    model = run_training(model, mk(itr), mk(iva), epochs=40, lr=1e-3, name="daily")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "smu": smu, "ssd": ssd,
                "seq_cols": DAILY_SEQ_COLS, "static_cols": STATIC_COLS,
                "seq_len": seq_len, "n_out": n_out}, OUT / "daily_lstm.pt")

    p, m, yp, ym = predict(model, mk(ite))
    rows, pred_rows = [], []
    for j, h in enumerate(HORIZONS):
        rows.append({"h": h, "model": "LSTM", "auc": roc_auc_score(yp[:, j], p[:, j]),
                     "brier": brier_score_loss(yp[:, j], p[:, j]),
                     "mae_mm": mean_absolute_error(ym[:, j], m[:, j])})
    for k, (c, t) in enumerate(ite):
        for j, h in enumerate(HORIZONS):
            pred_rows.append((cities[c], times[c][t], h, p[k, j], m[k, j]))
    pd.DataFrame(rows).to_csv(REPORT_DIR / "lstm_daily_metrics.csv", index=False)
    pd.DataFrame(pred_rows, columns=["city", "date", "h", "p_LSTM", "mm_LSTM"]).to_parquet(
        REPORT_DIR / "lstm_daily_test_predictions.parquet", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# ------------------------------------------------------------------ hourly
def train_hourly():
    arrays, statics, tp, tm, times = [], [], [], [], []
    for name, meta in TRAIN_CITIES.items():
        path = RAW_DIR / f"{name.lower()}_hourly.parquet"
        if not path.exists():
            continue
        h = pd.read_parquet(path)
        hp = prepare_hourly(h)
        arrays.append(hp[HOURLY_SEQ_COLS].to_numpy("float32"))
        statics.append(np.array([meta["lat"], meta["lon"], float(h["elevation"].iloc[0])], "float32"))
        mm = hp["precipitation"].to_numpy("float32")
        tm.append(mm)
        tp.append(np.where(np.isnan(mm), np.nan, (mm >= WET_HOUR_MM)).astype("float32"))
        times.append(hp["time"].to_numpy())
    seq_len, n_out = 72, MAX_LEAD
    years_tr, years_va = set(range(2014, 2023)), {2023}
    years_te = set(pd.DatetimeIndex(np.concatenate(times)).year) - years_tr - years_va
    itr = valid_index(arrays, tp, times, years_tr, seq_len, n_out, stride=3)
    iva = valid_index(arrays, tp, times, years_va, seq_len, n_out, stride=6)
    ite = valid_index(arrays, tp, times, years_te, seq_len, n_out, stride=3)
    mu, sd = scaler_fit(arrays, itr, seq_len)
    arrays_s = [np.nan_to_num((a - mu) / sd).astype("float32") for a in arrays]
    smu = np.mean(statics, 0)
    ssd = np.std(statics, 0) + 1e-6
    statics_s = [((s - smu) / ssd).astype("float32") for s in statics]
    mk = lambda ix: WindowDataset(arrays_s, statics_s, tp, tm, ix, seq_len, n_out)
    print(f"hourly LSTM windows: train {len(itr):,} val {len(iva):,} test {len(ite):,}")

    model = SeqModel(len(HOURLY_SEQ_COLS), len(STATIC_COLS), n_out, hidden=128).to(DEVICE)
    model = run_training(model, mk(itr), mk(iva), epochs=25, lr=1e-3, name="hourly")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "mu": mu, "sd": sd, "smu": smu, "ssd": ssd,
                "seq_cols": HOURLY_SEQ_COLS, "static_cols": STATIC_COLS,
                "seq_len": seq_len, "n_out": n_out, "hidden": 128}, OUT / "hourly_lstm.pt")

    p, m, yp, ym = predict(model, mk(ite))
    blocks = {"1-6h": range(0, 6), "7-12h": range(6, 12), "13-24h": range(12, 24), "25-48h": range(24, 48)}
    rows = []
    for b, cols in blocks.items():
        cols = list(cols)
        rows.append({"block": b, "model": "LSTM",
                     "auc": roc_auc_score(yp[:, cols].ravel(), p[:, cols].ravel()),
                     "brier": brier_score_loss(yp[:, cols].ravel(), p[:, cols].ravel()),
                     "mae_mm": mean_absolute_error(ym[:, cols].ravel(), m[:, cols].ravel())})
    pd.DataFrame(rows).to_csv(REPORT_DIR / "lstm_hourly_metrics.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("daily", "both"):
        train_daily()
    if which in ("hourly", "both"):
        train_hourly()
