"""
plot_stable_narx_rollouts.py

Plot rollouts of the saved stable NARX ridge model.

Outputs:
    identified_models/rollout_plots/*.png
    identified_models/rollout_plots/rollout_metrics.csv
"""

from __future__ import annotations

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    num = np.linalg.norm(y_true - y_pred, axis=0)
    den = np.linalg.norm(y_true - np.mean(y_true, axis=0, keepdims=True), axis=0)
    den = np.maximum(den, 1e-12)
    return 100.0 * (1.0 - num / den)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def load_run(path: Path, state_mode: str):
    df = pd.read_csv(path)

    if state_mode == "bend":
        y = df[["bx_rad", "by_rad"]].to_numpy(float)
    else:
        y = df[["phi_rad", "theta_rad"]].to_numpy(float)
        y[:, 0] = np.unwrap(y[:, 0])

    u = df[["T1_N", "T2_N", "T3_N"]].to_numpy(float)
    t = df["time_s"].to_numpy(float)
    return t, y, u


def standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    return (X - mean) / scale


def predict_delta(X: np.ndarray, W: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    Xn = standardize_apply(X, mean, scale)
    return Xn @ W


def make_one_feature(y_hist_full, u, k, ny, nu, ndu):
    du = np.zeros_like(u)
    du[1:] = u[1:] - u[:-1]

    dy = np.zeros_like(y_hist_full)
    dy[1:k+1] = y_hist_full[1:k+1] - y_hist_full[:k]

    feat = [1.0]

    for lag in range(ny + 1):
        feat.extend(y_hist_full[k - lag])

    for lag in range(1, ny + 1):
        feat.extend(dy[k - lag + 1])

    for lag in range(nu + 1):
        feat.extend(u[k - lag])

    for lag in range(ndu + 1):
        feat.extend(du[k - lag])

    return np.asarray(feat, float).reshape(1, -1)


def rollout(y, u, W, mean, scale, ny, nu, ndu, delta_clip, y_clip=2.0):
    n = len(y)
    max_lag = max(ny, nu, ndu + 1)

    y_pred = np.full_like(y, np.nan)
    y_pred[:max_lag + 1] = y[:max_lag + 1]

    for k in range(max_lag, n - 1):
        Xk = make_one_feature(y_pred, u, k, ny, nu, ndu)
        d = predict_delta(Xk, W, mean, scale).reshape(-1)
        d = np.clip(d, -delta_clip, delta_clip)
        y_pred[k + 1] = y_pred[k] + d
        y_pred[k + 1] = np.clip(y_pred[k + 1], -y_clip, y_clip)

    valid = np.arange(max_lag + 1, n)
    return valid, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/narx/stable_narx_ridge_bend.npz")
    parser.add_argument("--runs", type=str, default="data/selected/*.csv")
    parser.add_argument("--out-dir", type=str, default="models/narx/rollout_plots")
    args = parser.parse_args()

    model_path = Path(args.model)
    data = np.load(model_path, allow_pickle=True)

    W = data["W"]
    mean = data["feature_mean"]
    scale = data["feature_scale"]
    ny = int(data["ny"])
    nu = int(data["nu"])
    ndu = int(data["ndu"])
    state_mode = str(data["state_mode"])
    delta_clip = data["delta_clip"].astype(float)

    print("Loaded model:", model_path)
    print(f"state_mode={state_mode}, ny={ny}, nu={nu}, ndu={ndu}")
    print("delta_clip:", delta_clip)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    paths = [Path(p) for p in sorted(glob.glob(args.runs))]
    if not paths:
        raise FileNotFoundError(f"No files found for pattern: {args.runs}")

    for path in paths:
        t, y, u = load_run(path, state_mode)
        valid, y_pred = rollout(y, u, W, mean, scale, ny, nu, ndu, delta_clip)

        y_true_v = y[valid]
        y_pred_v = y_pred[valid]
        t_v = t[valid] - t[valid[0]]

        f = fit_percent(y_true_v, y_pred_v)
        e = rmse(y_true_v, y_pred_v)

        rows.append({
            "file": path.name,
            "fit_bx_percent": f[0],
            "fit_by_percent": f[1],
            "fit_mean_percent": float(np.mean(f)),
            "rmse_bx_rad": e[0],
            "rmse_by_rad": e[1],
            "rmse_bx_deg": np.rad2deg(e[0]),
            "rmse_by_deg": np.rad2deg(e[1]),
        })

        # Time plot bx/by
        plt.figure(figsize=(12, 7))
        plt.plot(t_v, np.rad2deg(y_true_v[:, 0]), label="measured bx")
        plt.plot(t_v, np.rad2deg(y_pred_v[:, 0]), "--", label="predicted bx")
        plt.plot(t_v, np.rad2deg(y_true_v[:, 1]), label="measured by")
        plt.plot(t_v, np.rad2deg(y_pred_v[:, 1]), "--", label="predicted by")
        plt.xlabel("time [s]")
        plt.ylabel("bend component [deg]")
        plt.title(f"Rollout: {path.name}\nfit bx={f[0]:.1f}%, by={f[1]:.1f}%")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_dir / f"{path.stem}_time_rollout.png", dpi=200, bbox_inches="tight")
        plt.close()

        # Bend-space plot
        plt.figure(figsize=(7, 7))
        plt.plot(np.rad2deg(y_true_v[:, 0]), np.rad2deg(y_true_v[:, 1]), ".", markersize=2, label="measured")
        plt.plot(np.rad2deg(y_pred_v[:, 0]), np.rad2deg(y_pred_v[:, 1]), ".", markersize=2, label="predicted")
        plt.xlabel("bx [deg]")
        plt.ylabel("by [deg]")
        plt.title(f"Bend-space rollout: {path.name}")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_dir / f"{path.stem}_bendspace_rollout.png", dpi=200, bbox_inches="tight")
        plt.close()

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / "rollout_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print("\nRollout metrics:")
    print(metrics.to_string(index=False))
    print("\nSaved plots to:", out_dir)
    print("Saved metrics:", metrics_path)


if __name__ == "__main__":
    main()
