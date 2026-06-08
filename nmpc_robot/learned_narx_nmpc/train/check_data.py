"""
check_sysid_dataset.py
It prints:
  - number of samples
  - duration and average dt
  - theta range
  - bend-space range
  - tension/current ranges
  - whether NaN/Inf values exist

It also saves:
  - sysid_dataset_summary.csv
  - sysid_bend_coverage.png
  - sysid_tension_coverage.png
"""

from __future__ import annotations

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

import argparse
from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = [
    "time_s",
    "phi_rad",
    "theta_rad",
    "phi_dot_rad_s",
    "theta_dot_rad_s",
    "bx_rad",
    "by_rad",
    "T1_N",
    "T2_N",
    "T3_N",
    "I1_mA",
    "I2_mA",
    "I3_mA",
]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return df


def summarize(path: Path, df: pd.DataFrame) -> dict:
    t = df["time_s"].to_numpy(float)
    dt = np.diff(t)

    theta_deg = np.rad2deg(df["theta_rad"].to_numpy(float))
    bx = df["bx_rad"].to_numpy(float)
    by = df["by_rad"].to_numpy(float)
    bend_norm_deg = np.rad2deg(np.sqrt(bx**2 + by**2))

    finite = np.isfinite(df[REQUIRED_COLUMNS].to_numpy(float)).all()

    return {
        "file": path.name,
        "samples": len(df),
        "duration_s": float(t[-1] - t[0]) if len(t) > 1 else 0.0,
        "mean_dt_s": float(np.mean(dt)) if len(dt) > 0 else np.nan,
        "max_dt_s": float(np.max(dt)) if len(dt) > 0 else np.nan,
        "finite_ok": bool(finite),
        "theta_min_deg": float(np.min(theta_deg)),
        "theta_max_deg": float(np.max(theta_deg)),
        "theta_mean_deg": float(np.mean(theta_deg)),
        "bend_norm_max_deg": float(np.max(bend_norm_deg)),
        "bx_min_rad": float(np.min(bx)),
        "bx_max_rad": float(np.max(bx)),
        "by_min_rad": float(np.min(by)),
        "by_max_rad": float(np.max(by)),
        "T1_min_N": float(df["T1_N"].min()),
        "T1_max_N": float(df["T1_N"].max()),
        "T2_min_N": float(df["T2_N"].min()),
        "T2_max_N": float(df["T2_N"].max()),
        "T3_min_N": float(df["T3_N"].min()),
        "T3_max_N": float(df["T3_N"].max()),
        "max_abs_current_mA": float(np.max(np.abs(df[["I1_mA", "I2_mA", "I3_mA"]].to_numpy(float)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, default="data/selected/*.csv")
    parser.add_argument("--out-dir", type=str, default="data/analysis")
    args = parser.parse_args()

    paths = [Path(p) for p in sorted(glob.glob(args.runs))]
    if not paths:
        raise FileNotFoundError(f"No CSV files found for pattern: {args.runs}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    rows = []

    for path in paths:
        df = read_csv(path)
        dfs.append((path, df))
        rows.append(summarize(path, df))

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "sysid_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n========== SYSID DATASET SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")

    # Bend coverage plot
    plt.figure(figsize=(7, 7))
    for path, df in dfs:
        plt.plot(np.rad2deg(df["bx_rad"]), np.rad2deg(df["by_rad"]), ".", markersize=2, label=path.stem)
    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)
    plt.xlabel("bx = theta cos(phi) [deg]")
    plt.ylabel("by = theta sin(phi) [deg]")
    plt.title("Bend-space coverage")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(fontsize=8)
    bend_path = out_dir / "sysid_bend_coverage.png"
    plt.savefig(bend_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Tension coverage plot
    plt.figure(figsize=(8, 6))
    for path, df in dfs:
        plt.plot(df["T1_N"], df["T2_N"], ".", markersize=2, label=f"{path.stem}: T1/T2")
        plt.plot(df["T1_N"], df["T3_N"], ".", markersize=2, label=f"{path.stem}: T1/T3")
    plt.xlabel("T1 [N]")
    plt.ylabel("T2 or T3 [N]")
    plt.title("Tension command coverage")
    plt.grid(True)
    plt.legend(fontsize=7)
    tension_path = out_dir / "sysid_tension_coverage.png"
    plt.savefig(tension_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved bend plot: {bend_path}")
    print(f"Saved tension plot: {tension_path}")

    print("\n========== WARNINGS / CHECKS ==========")
    any_warn = False
    for row in rows:
        if not row["finite_ok"]:
            print(f"[WARN] {row['file']} contains NaN or Inf.")
            any_warn = True
        if row["theta_max_deg"] < 10:
            print(f"[WARN] {row['file']} has weak bending: theta max < 10 deg.")
            any_warn = True
        if row["max_abs_current_mA"] > 250:
            print(f"[WARN] {row['file']} reached high current: {row['max_abs_current_mA']:.1f} mA.")
            any_warn = True
        if row["max_dt_s"] > 0.25:
            print(f"[WARN] {row['file']} has a large timing gap: max dt = {row['max_dt_s']:.3f} s.")
            any_warn = True

    if not any_warn:
        print("No obvious problems found. You can train ARX/SINDYc next.")


if __name__ == "__main__":
    main()
