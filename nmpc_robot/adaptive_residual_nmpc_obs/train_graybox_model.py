from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from graybox_model import AdaptiveResidualModel, ridge_fit, sparsify_ridge
from params_graybox import data_collection, model as model_cfg, mpc


FEATURE_NAMES = np.array([
    "1",
    "bx", "by", "vx", "vy", "mx", "my",
    "dT1", "dT2", "dT3",
    "Teff1_minus_pre", "Teff2_minus_pre", "Teff3_minus_pre",
    "tanh_3vx", "tanh_3vy",
    "bx_vx", "by_vy",
    "bx_mx", "bx_my", "by_mx", "by_my",
    "vx_mx", "vx_my", "vy_mx", "vy_my",
    "bx2", "by2", "vx2", "vy2", "mx2", "my2", "mx_my",
], dtype=object)


def load_csvs(data_path: str | Path) -> list[pd.DataFrame]:
    p = Path(data_path)
    if p.is_file():
        return [pd.read_csv(p)]
    files = sorted(p.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {p}")
    return [pd.read_csv(f) for f in files]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of these columns found: {candidates}. Available: {list(df.columns)}")


def smooth_series(x: np.ndarray, window: int, poly: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < max(window, poly + 2):
        return x.copy()
    window = min(int(window), len(x) - 1 if len(x) % 2 == 0 else len(x))
    if window % 2 == 0:
        window -= 1
    if window < poly + 2:
        return x.copy()
    return savgol_filter(x, window_length=window, polyorder=poly, axis=0, mode="interp")


def tendon_moment_from_angles(tensions: np.ndarray, pretension: float, angles_rad: np.ndarray, radius: float) -> np.ndarray:
    T = np.asarray(tensions, dtype=float)
    dT = T - float(pretension)
    dT = dT - np.mean(dT, axis=1, keepdims=True)
    mx = float(radius) * (dT @ np.cos(angles_rad))
    my = float(radius) * (dT @ np.sin(angles_rad))
    return np.column_stack([mx, my])


def compute_teff(u: np.ndarray, alpha_u: float) -> np.ndarray:
    teff = np.zeros_like(u, dtype=float)
    teff[0] = u[0]
    for k in range(1, len(u)):
        teff[k] = teff[k - 1] + float(alpha_u) * (u[k] - teff[k - 1])
    return teff


def build_residual_features(b: np.ndarray, v: np.ndarray, u: np.ndarray, teff: np.ndarray, du: np.ndarray, m: np.ndarray, pretension: float) -> np.ndarray:
    bx = b[:, 0]
    by = b[:, 1]
    vx = v[:, 0]
    vy = v[:, 1]
    mx = m[:, 0]
    my = m[:, 1]
    ue = teff - float(pretension)
    return np.column_stack([
        np.ones(len(b)),
        bx, by, vx, vy, mx, my,
        du[:, 0], du[:, 1], du[:, 2],
        ue[:, 0], ue[:, 1], ue[:, 2],
        np.tanh(3.0 * vx), np.tanh(3.0 * vy),
        bx * vx, by * vy,
        bx * mx, bx * my, by * mx, by * my,
        vx * mx, vx * my, vy * mx, vy * my,
        bx * bx, by * by, vx * vx, vy * vy, mx * mx, my * my, mx * my,
    ])


def extract_training_arrays(
    frames: list[pd.DataFrame],
    dt_default: float,
    pretension: float,
    tendon_angles_rad: np.ndarray,
    tendon_radius_m: float,
    tendon_lag_tau: float,
    smooth_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    X_phys_all = []
    X_res_all = []
    A_all = []
    file_id_all = []
    dt_list = []

    alpha_u = float(dt_default / max(float(tendon_lag_tau), float(dt_default)))
    alpha_u = float(np.clip(alpha_u, 0.0, 1.0))

    for file_idx, df in enumerate(frames):
        bx_col = find_col(df, ["bx_rad", "bx"])
        by_col = find_col(df, ["by_rad", "by"])
        t1_col = find_col(df, ["T1_N", "T1", "u1"])
        t2_col = find_col(df, ["T2_N", "T2", "u2"])
        t3_col = find_col(df, ["T3_N", "T3", "u3"])

        if "time_s" in df.columns and len(df) > 3:
            t = df["time_s"].to_numpy(float)
            dt = float(np.median(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                dt = float(dt_default)
        else:
            dt = float(dt_default)
        dt_list.append(dt)

        b_raw = df[[bx_col, by_col]].to_numpy(float)
        u = df[[t1_col, t2_col, t3_col]].to_numpy(float)

        b = smooth_series(b_raw, window=smooth_window, poly=3)
        v = np.gradient(b, dt, axis=0)
        a = np.gradient(v, dt, axis=0)

        teff = compute_teff(u, alpha_u)
        du = np.zeros_like(u)
        du[1:] = u[1:] - u[:-1]
        m = tendon_moment_from_angles(teff, pretension, tendon_angles_rad, tendon_radius_m)

        X_phys = np.column_stack([np.ones(len(b)), b[:, 0], b[:, 1], v[:, 0], v[:, 1], m[:, 0], m[:, 1]])
        X_res = build_residual_features(b, v, u, teff, du, m, pretension)

        margin = max(3, smooth_window // 2)
        sl = slice(margin, max(margin + 1, len(b) - margin))
        X_phys_all.append(X_phys[sl])
        X_res_all.append(X_res[sl])
        A_all.append(a[sl])
        file_id_all.append(np.full(len(a[sl]), file_idx, dtype=int))

    X_phys_all = np.vstack(X_phys_all)
    X_res_all = np.vstack(X_res_all)
    A_all = np.vstack(A_all)
    file_id_all = np.concatenate(file_id_all)
    dt_median = float(np.median(dt_list)) if dt_list else float(dt_default)
    return X_phys_all, X_res_all, A_all, file_id_all, dt_median


def nrmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = np.linalg.norm(y - np.mean(y, axis=0, keepdims=True))
    if denom < 1e-12:
        return float("nan")
    return float(100.0 * (1.0 - np.linalg.norm(y - yhat) / denom))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=data_collection["out_dir"], help="CSV file or folder containing collection CSVs")
    parser.add_argument("--out", default=model_cfg["path"], help="Output .npz model path")
    parser.add_argument("--dt", type=float, default=float(data_collection["dt"]))
    parser.add_argument("--pretension", type=float, default=float(mpc["pretension"]))
    parser.add_argument("--tendon-lag-tau", type=float, default=float(model_cfg["tendon_lag_tau"]))
    parser.add_argument("--tendon-radius-m", type=float, default=float(model_cfg["tendon_radius_m"]))
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--phys-ridge", type=float, default=1e-4)
    parser.add_argument("--res-ridge", type=float, default=2e-4)
    parser.add_argument("--res-threshold", type=float, default=2e-3)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--accel-clip", type=float, default=60.0)
    parser.add_argument("--max-bend-deg", type=float, default=85.0)
    parser.add_argument("--max-bend-dot-deg-s", type=float, default=500.0)
    args = parser.parse_args()

    frames = load_csvs(args.data)
    tendon_angles_rad = np.deg2rad(np.asarray(model_cfg["tendon_angles_deg"], dtype=float))

    Xp, Xr, A, file_id, dt_median = extract_training_arrays(
        frames=frames,
        dt_default=args.dt,
        pretension=args.pretension,
        tendon_angles_rad=tendon_angles_rad,
        tendon_radius_m=args.tendon_radius_m,
        tendon_lag_tau=args.tendon_lag_tau,
        smooth_window=args.smooth_window,
    )

    unique_files = np.unique(file_id)
    if len(unique_files) >= 3:
        val_files = unique_files[::5] if len(unique_files) >= 5 else unique_files[-1:]
        val_mask = np.isin(file_id, val_files)
    else:
        idx = np.arange(len(A))
        val_mask = idx >= int(0.8 * len(idx))
    train_mask = ~val_mask

    W_phys = ridge_fit(Xp[train_mask], A[train_mask], args.phys_ridge)
    A_phys_train = Xp[train_mask] @ W_phys
    A_phys_val = Xp[val_mask] @ W_phys

    R_train = A[train_mask] - A_phys_train
    W_res = sparsify_ridge(Xr[train_mask], R_train, lam=args.res_ridge, threshold=args.res_threshold, n_iter=8)

    A_hat_train = A_phys_train + args.residual_scale * (Xr[train_mask] @ W_res)
    A_hat_val = A_phys_val + args.residual_scale * (Xr[val_mask] @ W_res)

    print("============================================================")
    print("Adaptive residual gray-box model training")
    print("CSV files:", len(frames))
    print("Samples:", len(A), "train:", int(np.sum(train_mask)), "val:", int(np.sum(val_mask)))
    print("dt median:", dt_median)
    print("Physics acceleration fit train/val [%]:", round(nrmse(A[train_mask], A_phys_train), 2), round(nrmse(A[val_mask], A_phys_val), 2))
    print("Gray-box acceleration fit train/val [%]:", round(nrmse(A[train_mask], A_hat_train), 2), round(nrmse(A[val_mask], A_hat_val), 2))
    print("Nonzero residual terms:", int(np.count_nonzero(W_res)), "/", W_res.size)
    print("============================================================")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    alpha_u = float(args.dt / max(float(args.tendon_lag_tau), float(args.dt)))
    alpha_u = float(np.clip(alpha_u, 0.0, 1.0))

    np.savez(
        out_path,
        dt=float(args.dt),
        pretension=float(args.pretension),
        tendon_angles_rad=tendon_angles_rad,
        tendon_radius_m=float(args.tendon_radius_m),
        tendon_lag_tau=float(args.tendon_lag_tau),
        alpha_u=alpha_u,
        W_phys=W_phys,
        W_res=W_res,
        residual_scale=float(args.residual_scale),
        accel_clip=float(args.accel_clip),
        max_bend_rad=float(np.deg2rad(args.max_bend_deg)),
        max_bend_dot_rad_s=float(np.deg2rad(args.max_bend_dot_deg_s)),
        feature_names=FEATURE_NAMES,
        train_nrmse_phys=nrmse(A[train_mask], A_phys_train),
        val_nrmse_phys=nrmse(A[val_mask], A_phys_val),
        train_nrmse_graybox=nrmse(A[train_mask], A_hat_train),
        val_nrmse_graybox=nrmse(A[val_mask], A_hat_val),
    )
    print("saved:", out_path)

    _ = AdaptiveResidualModel(out_path)


if __name__ == "__main__":
    main()
