from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graybox_model import AdaptiveResidualModel, MpcSettings
from params_graybox import data_collection, model as model_cfg, mpc, safety


def load_csvs(data_path: str | Path) -> list[tuple[Path, pd.DataFrame]]:
    p = Path(data_path)
    if p.is_file():
        return [(p, pd.read_csv(p))]
    files = sorted(p.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {p}")
    return [(f, pd.read_csv(f)) for f in files]


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")


def nrmse(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.linalg.norm(y - np.mean(y, axis=0, keepdims=True))
    if denom < 1e-12:
        return float("nan")
    return float(100.0 * (1.0 - np.linalg.norm(y - yhat) / denom))


def rollout_one(model: AdaptiveResidualModel, df: pd.DataFrame, settings: MpcSettings, reset_every: int | None = None):
    bx_col = find_col(df, ["bx_rad", "bx"])
    by_col = find_col(df, ["by_rad", "by"])
    t1_col = find_col(df, ["T1_N", "T1", "u1"])
    t2_col = find_col(df, ["T2_N", "T2", "u2"])
    t3_col = find_col(df, ["T3_N", "T3", "u3"])

    b = df[[bx_col, by_col]].to_numpy(float)
    u = df[[t1_col, t2_col, t3_col]].to_numpy(float)
    if len(b) < 3:
        return b, b.copy()

    x = model.make_state(b[0], u[0])
    yhat = np.zeros_like(b)
    yhat[0] = b[0]
    prev_u = u[0]
    for k in range(1, len(b)):
        if reset_every is not None and reset_every > 0 and k % reset_every == 0:
            x = model.make_state(b[k - 1], u[k - 1])
            prev_u = u[k - 1]
        dU = u[k] - prev_u
        x = model.step(x, dU, settings, use_disturbance=False)
        yhat[k] = x[0:2]
        prev_u = u[k]
    return b, yhat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=model_cfg["path"])
    parser.add_argument("--data", default=data_collection["out_dir"])
    parser.add_argument("--out-dir", default="validation_graybox")
    parser.add_argument("--reset-every", type=int, default=6, help="reset open-loop prediction every N samples; 0 means one long rollout. Use 6 to match the current NMPC horizon.")
    args = parser.parse_args()

    model = AdaptiveResidualModel(args.model)
    settings = MpcSettings(
        horizon=int(mpc["horizon"]),
        u_min=float(mpc["u_min"]),
        u_max=float(mpc["u_max"]),
        max_du_step=float(mpc["max_du_step"]),
        qy=float(mpc["qy"]),
        qf=float(mpc["qf"]),
        rdu=float(mpc["rdu"]),
        ru=float(mpc["ru"]),
        qtheta=float(mpc.get("qtheta", 0.0)),
        qdist=float(mpc.get("qdist", 0.0)),
        u_bias=np.asarray(mpc["initial_bias"], dtype=float),
        max_bend_deg=float(safety["max_theta_deg"]),
    )
    reset_every = None if args.reset_every <= 0 else int(args.reset_every)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path, df in load_csvs(args.data):
        b, yhat = rollout_one(model, df, settings, reset_every=reset_every)
        err_deg = np.rad2deg(np.linalg.norm(b - yhat, axis=1))
        rows.append({
            "file": str(path),
            "samples": len(b),
            "nrmse_percent": nrmse(b, yhat),
            "mean_error_deg": float(np.mean(err_deg)),
            "p95_error_deg": float(np.percentile(err_deg, 95)),
            "max_error_deg": float(np.max(err_deg)),
        })

        stem = path.stem.replace(" ", "_")[-80:]
        t = df["time_s"].to_numpy(float) if "time_s" in df.columns else np.arange(len(b)) * model.dt
        plt.figure(figsize=(10, 5))
        plt.plot(t, np.rad2deg(b[:, 0]), label="bx measured")
        plt.plot(t, np.rad2deg(b[:, 1]), label="by measured")
        plt.plot(t, np.rad2deg(yhat[:, 0]), "--", label="bx predicted")
        plt.plot(t, np.rad2deg(yhat[:, 1]), "--", label="by predicted")
        plt.xlabel("time [s]")
        plt.ylabel("bend [deg]")
        plt.title(f"Gray-box rollout: {path.name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_rollout.png", dpi=180)
        plt.close()

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print("saved:", out_dir / "summary.csv")


if __name__ == "__main__":
    main()
