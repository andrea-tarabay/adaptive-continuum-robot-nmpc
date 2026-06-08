"""
train_stable_narx_ridge.py

Stable delayed NARX baseline for tendon-driven soft arm system ID.

Usage:
    python train_stable_narx_ridge.py --runs "sysid_runs/*.csv" --state-mode bend --ny 6 --nu 6 --ndu 3
    python train_stable_narx_ridge.py --runs "sysid_runs/*.csv" --state-mode bend --ny 10 --nu 10 --ndu 4

Outputs:
    identified_models/stable_narx_ridge_bend.npz
    identified_models/stable_narx_ridge_validation_summary.csv
"""

from __future__ import annotations

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

import argparse
import glob
from pathlib import Path
import json

import numpy as np
import pandas as pd


def fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.linalg.norm(y_true - y_pred, axis=0)
    den = np.linalg.norm(y_true - np.mean(y_true, axis=0, keepdims=True), axis=0)
    den = np.maximum(den, 1e-12)
    return 100.0 * (1.0 - num / den)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def read_run(path: Path, state_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)

    required = ["time_s", "phi_rad", "theta_rad", "bx_rad", "by_rad", "T1_N", "T2_N", "T3_N"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    finite = np.isfinite(df[required].to_numpy(float)).all(axis=1)
    df = df.loc[finite].reset_index(drop=True)

    t = df["time_s"].to_numpy(float)

    if state_mode == "bend":
        y = df[["bx_rad", "by_rad"]].to_numpy(float)
    elif state_mode == "phi_theta":
        y = df[["phi_rad", "theta_rad"]].to_numpy(float)
        y[:, 0] = np.unwrap(y[:, 0])
    else:
        raise ValueError(state_mode)

    u = df[["T1_N", "T2_N", "T3_N"]].to_numpy(float)
    return t, y, u


def make_feature_names(ny: int, nu: int, ndu: int) -> list[str]:
    names = ["1"]

    for lag in range(ny + 1):
        suf = "" if lag == 0 else f"_m{lag}"
        names += [f"bx{suf}", f"by{suf}"]

    for lag in range(1, ny + 1):
        suf = "" if lag == 1 else f"_m{lag-1}"
        names += [f"dbx{suf}", f"dby{suf}"]

    for lag in range(nu + 1):
        suf = "" if lag == 0 else f"_m{lag}"
        names += [f"T1{suf}", f"T2{suf}", f"T3{suf}"]

    for lag in range(ndu + 1):
        suf = "" if lag == 0 else f"_m{lag}"
        names += [f"dT1{suf}", f"dT2{suf}", f"dT3{suf}"]

    return names


def build_features(y: np.ndarray, u: np.ndarray, ny: int, nu: int, ndu: int):
    """
    Build regression rows for predicting residual:
        target = y[k+1] - y[k]
    """
    n = len(y)
    du = np.zeros_like(u)
    du[1:] = u[1:] - u[:-1]

    dy = np.zeros_like(y)
    dy[1:] = y[1:] - y[:-1]

    max_lag = max(ny, nu, ndu + 1)
    X = []
    Y_delta = []
    K = []

    for k in range(max_lag, n - 1):
        feat = [1.0]

        # y history
        for lag in range(ny + 1):
            feat.extend(y[k - lag])

        for lag in range(1, ny + 1):
            feat.extend(dy[k - lag + 1])

        # u history
        for lag in range(nu + 1):
            feat.extend(u[k - lag])

        # du history
        for lag in range(ndu + 1):
            feat.extend(du[k - lag])

        X.append(feat)
        Y_delta.append(y[k + 1] - y[k])
        K.append(k)

    return np.asarray(X, float), np.asarray(Y_delta, float), np.asarray(K, int)


def standardize_train(X: np.ndarray):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    mean[0] = 0.0
    scale[0] = 1.0
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (X - mean) / scale, mean, scale


def standardize_apply(X: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    return (X - mean) / scale


def fit_ridge(Xn: np.ndarray, Y: np.ndarray, alpha: float):
    nfeat = Xn.shape[1]
    A = Xn.T @ Xn + alpha * np.eye(nfeat)
    A[0, 0] -= alpha
    B = Xn.T @ Y
    return np.linalg.solve(A, B)


def predict_delta(X: np.ndarray, W: np.ndarray, mean: np.ndarray, scale: np.ndarray):
    Xn = standardize_apply(X, mean, scale)
    return Xn @ W


def one_step_validate(y, u, W, mean, scale, ny, nu, ndu):
    X, Ydelta, K = build_features(y, u, ny, nu, ndu)
    d_hat = predict_delta(X, W, mean, scale)
    y_true = y[K + 1]
    y_pred = y[K] + d_hat
    return y_true, y_pred


def make_one_feature(y_hist_full, u, k, ny, nu, ndu):
    """
    Build one feature row during rollout.
    y_hist_full contains predicted/seeded y values up to current k.
    """
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


def rollout_validate(y, u, W, mean, scale, ny, nu, ndu, delta_clip=None, y_clip=2.0):
    n = len(y)
    max_lag = max(ny, nu, ndu + 1)

    y_pred = np.full_like(y, np.nan)
    y_pred[:max_lag + 1] = y[:max_lag + 1]

    for k in range(max_lag, n - 1):
        Xk = make_one_feature(y_pred, u, k, ny, nu, ndu)
        d = predict_delta(Xk, W, mean, scale).reshape(-1)

        if delta_clip is not None:
            d = np.clip(d, -delta_clip, delta_clip)

        y_pred[k + 1] = y_pred[k] + d
        y_pred[k + 1] = np.clip(y_pred[k + 1], -y_clip, y_clip)

    valid = np.arange(max_lag + 1, n)
    return y[valid], y_pred[valid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default="data/selected/*.csv")
    parser.add_argument("--state-mode", choices=["bend", "phi_theta"], default="bend")
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nu", type=int, default=6)
    parser.add_argument("--ndu", type=int, default=3)
    parser.add_argument("--ridge-grid", type=float, nargs="+",
                        default=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    parser.add_argument("--clip-factor", type=float, default=3.0,
                        help="Rollout delta clip = clip_factor * 99th percentile abs training delta.")
    parser.add_argument("--no-delta-clip", action="store_true")
    parser.add_argument("--out-dir", type=str, default="models/narx")
    args = parser.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.runs))]
    if len(paths) < 2:
        raise ValueError("Need at least two runs for leave-one-run-out validation.")

    runs = []
    for p in paths:
        t, y, u = read_run(p, args.state_mode)
        runs.append({"path": p, "t": t, "y": y, "u": u})

    print("ridge, mean_one_step_fit, mean_rollout_fit, one_step_per_output, rollout_per_output, rollout_rmse")

    results = []
    best = None

    for alpha in args.ridge_grid:
        one_fits = []
        roll_fits = []
        roll_rmses = []

        for val_idx, val_run in enumerate(runs):
            X_train_list = []
            Y_train_list = []

            for i, run in enumerate(runs):
                X_i, Y_i, _ = build_features(run["y"], run["u"], args.ny, args.nu, args.ndu)
                if i != val_idx:
                    X_train_list.append(X_i)
                    Y_train_list.append(Y_i)

            X_train = np.vstack(X_train_list)
            Y_train = np.vstack(Y_train_list)

            Xn, mean, scale = standardize_train(X_train)
            W = fit_ridge(Xn, Y_train, alpha)

            if args.no_delta_clip:
                delta_clip = None
            else:
                delta_clip = args.clip_factor * np.percentile(np.abs(Y_train), 99.0, axis=0)
                delta_clip = np.maximum(delta_clip, np.deg2rad(0.1))

            y1_true, y1_pred = one_step_validate(
                val_run["y"], val_run["u"], W, mean, scale,
                args.ny, args.nu, args.ndu
            )

            yr_true, yr_pred = rollout_validate(
                val_run["y"], val_run["u"], W, mean, scale,
                args.ny, args.nu, args.ndu,
                delta_clip=delta_clip,
            )

            one_fits.append(fit_percent(y1_true, y1_pred))
            roll_fits.append(fit_percent(yr_true, yr_pred))
            roll_rmses.append(rmse(yr_true, yr_pred))

        one_fits = np.vstack(one_fits)
        roll_fits = np.vstack(roll_fits)
        roll_rmses = np.vstack(roll_rmses)

        mean_one = float(np.mean(one_fits))
        mean_roll = float(np.mean(roll_fits))
        per_one = np.mean(one_fits, axis=0)
        per_roll = np.mean(roll_fits, axis=0)
        per_rmse = np.mean(roll_rmses, axis=0)

        print(f"{alpha:g}, {mean_one:.2f}, {mean_roll:.2f}, {per_one}, {per_roll}, {per_rmse}")

        row = {
            "ridge": alpha,
            "mean_one_step_fit": mean_one,
            "mean_rollout_fit": mean_roll,
            "one_step_fit_bx": per_one[0],
            "one_step_fit_by": per_one[1],
            "rollout_fit_bx": per_roll[0],
            "rollout_fit_by": per_roll[1],
            "rollout_rmse_bx": per_rmse[0],
            "rollout_rmse_by": per_rmse[1],
        }
        results.append(row)

        if best is None or mean_roll > best["mean_rollout_fit"]:
            best = row

    assert best is not None
    best_alpha = float(best["ridge"])

    # Final model on all data
    X_all_list = []
    Y_all_list = []
    for run in runs:
        X_i, Y_i, _ = build_features(run["y"], run["u"], args.ny, args.nu, args.ndu)
        X_all_list.append(X_i)
        Y_all_list.append(Y_i)

    X_all = np.vstack(X_all_list)
    Y_all = np.vstack(Y_all_list)
    Xn_all, mean_all, scale_all = standardize_train(X_all)
    W_all = fit_ridge(Xn_all, Y_all, best_alpha)

    if args.no_delta_clip:
        delta_clip = None
    else:
        delta_clip = args.clip_factor * np.percentile(np.abs(Y_all), 99.0, axis=0)
        delta_clip = np.maximum(delta_clip, np.deg2rad(0.1))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = np.asarray(make_feature_names(args.ny, args.nu, args.ndu), dtype=object)
    model_path = out_dir / f"stable_narx_ridge_{args.state_mode}.npz"
    np.savez(
        model_path,
        W=W_all,
        feature_mean=mean_all,
        feature_scale=scale_all,
        feature_names=names,
        ny=args.ny,
        nu=args.nu,
        ndu=args.ndu,
        state_mode=args.state_mode,
        best_ridge=best_alpha,
        delta_clip=np.asarray(delta_clip if delta_clip is not None else [np.inf, np.inf], dtype=float),
        train_files=np.asarray([str(r["path"]) for r in runs], dtype=object),
        metadata=json.dumps(vars(args)),
    )

    summary_path = out_dir / "stable_narx_ridge_validation_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)

    print("\nBest ridge:", best_alpha)
    print("Best mean one-step fit:", f"{best['mean_one_step_fit']:.2f}%")
    print("Best mean rollout fit:", f"{best['mean_rollout_fit']:.2f}%")
    print("Saved:", model_path)
    print("Saved:", summary_path)

    print("\nLargest coefficients:")
    for out_idx, out_name in enumerate(["delta_bx", "delta_by"]):
        coefs = W_all[:, out_idx]
        order = np.argsort(-np.abs(coefs))
        print(f"  {out_name}:")
        for idx in order[:20]:
            print(f"    {coefs[idx]:+.5g} * {names[idx]}")


if __name__ == "__main__":
    main()
