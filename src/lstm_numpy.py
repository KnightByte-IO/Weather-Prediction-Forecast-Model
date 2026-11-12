"""Run the trained LSTMs with NumPy only, so the web app does not need PyTorch.

train_lstm.py saves each model twice: a PyTorch checkpoint (*.pt) and the same weights
as a NumPy archive (*.npz). This module reads the .npz and repeats the model's forward
pass (2-layer LSTM -> dense layer with ReLU -> rain-logit head and log1p(mm) head).

Export again from a checkpoint:  python src/lstm_numpy.py
"""
from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def export_npz(pt_path, npz_path=None):
    """Write the weights and scalers of a PyTorch checkpoint to a .npz file (needs torch)."""
    import torch

    pt_path = Path(pt_path)
    npz_path = Path(npz_path) if npz_path else pt_path.with_suffix(".npz")
    ck = torch.load(pt_path, weights_only=False)
    arrays = {f"w__{k}": v.detach().cpu().numpy() for k, v in ck["state"].items()}
    for k in ("mu", "sd", "smu", "ssd"):
        arrays[k] = np.asarray(ck[k], dtype="float32")
    arrays["seq_cols"] = np.array(ck["seq_cols"])
    arrays["static_cols"] = np.array(ck["static_cols"])
    arrays["seq_len"] = np.array(ck["seq_len"])
    arrays["n_out"] = np.array(ck["n_out"])
    np.savez_compressed(npz_path, **arrays)
    return npz_path


class NumpyLSTM:
    """Same forward pass as train_lstm.SeqModel in eval mode (dropout off)."""

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=False)
        self.w = {k[3:]: z[k].astype("float64") for k in z.files if k.startswith("w__")}
        self.layers = sum(1 for k in self.w if k.startswith("lstm.weight_ih_l"))
        self.meta = {
            "mu": z["mu"], "sd": z["sd"], "smu": z["smu"], "ssd": z["ssd"],
            "seq_cols": [str(c) for c in z["seq_cols"]],
            "static_cols": [str(c) for c in z["static_cols"]],
            "seq_len": int(z["seq_len"]), "n_out": int(z["n_out"]),
        }

    def __call__(self, seq, static):
        """seq: (T, n_in) scaled inputs; static: (n_static,) scaled. Returns (logits, log1p_mm)."""
        x = np.asarray(seq, dtype="float64")
        for layer in range(self.layers):
            w_ih, w_hh = self.w[f"lstm.weight_ih_l{layer}"], self.w[f"lstm.weight_hh_l{layer}"]
            b = self.w[f"lstm.bias_ih_l{layer}"] + self.w[f"lstm.bias_hh_l{layer}"]
            hidden = w_hh.shape[1]
            h, c = np.zeros(hidden), np.zeros(hidden)
            xw = x @ w_ih.T + b                      # input part of the gates for every step
            out = np.empty((len(x), hidden))
            for t in range(len(x)):
                g = xw[t] + w_hh @ h
                i, f = _sigmoid(g[:hidden]), _sigmoid(g[hidden:2 * hidden])
                cell, o = np.tanh(g[2 * hidden:3 * hidden]), _sigmoid(g[3 * hidden:])
                c = f * c + i * cell
                h = o * np.tanh(c)
                out[t] = h
            x = out
        zin = np.concatenate([h, np.asarray(static, dtype="float64")])
        zz = np.maximum(0.0, self.w["head.0.weight"] @ zin + self.w["head.0.bias"])
        logit = self.w["logit.weight"] @ zz + self.w["logit.bias"]
        mm = self.w["mm.weight"] @ zz + self.w["mm.bias"]
        return logit, mm


if __name__ == "__main__":
    from config import MODEL_DIR

    for name in ("daily", "hourly"):
        pt = MODEL_DIR / "lstm" / f"{name}_lstm.pt"
        if pt.exists():
            print("wrote", export_npz(pt))
