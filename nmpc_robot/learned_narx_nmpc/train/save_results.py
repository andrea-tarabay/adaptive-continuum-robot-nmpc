from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_run(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "log.csv"
    df.to_csv(csv_path, index=False)

    if len(df) == 0:
        return csv_path

    t = df["time_s"].to_numpy(float)

    plt.figure(figsize=(10, 6))
    plt.plot(t, df["bx_deg"], label="bx")
    plt.plot(t, df["by_deg"], label="by")
    plt.plot(t, df["target_bx_deg"], "--", label="target bx")
    plt.plot(t, df["target_by_deg"], "--", label="target by")
    plt.xlabel("time [s]")
    plt.ylabel("bend component [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bend_tracking.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(t, df["bend_error_deg"], label="bend error")
    plt.xlabel("time [s]")
    plt.ylabel("error [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "bend_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, df["T1_N"], label="T1")
    plt.plot(t, df["T2_N"], label="T2")
    plt.plot(t, df["T3_N"], label="T3")
    plt.xlabel("time [s]")
    plt.ylabel("tension [N]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "tensions.png", dpi=200)
    plt.close()

    return csv_path


def print_summary(rows: list[dict]):
    if not rows:
        print("No rows logged.")
        return

    df = pd.DataFrame(rows)
    last = df[df["time_s"] >= df["time_s"].max() - 5.0]

    print("\n================ summary, last 5 s ================")
    print("phi/theta [deg]:", round(float(last["phi_deg"].mean()), 2), round(float(last["theta_deg"].mean()), 2))
    print("bx/by [deg]:", round(float(last["bx_deg"].mean()), 2), round(float(last["by_deg"].mean()), 2))
    print("target bx/by [deg]:", round(float(last["target_bx_deg"].mean()), 2), round(float(last["target_by_deg"].mean()), 2))
    print("bend error [deg]:", round(float(last["bend_error_deg"].mean()), 2))
    print("tensions [N]:", round(float(last["T1_N"].mean()), 3), round(float(last["T2_N"].mean()), 3), round(float(last["T3_N"].mean()), 3))
    print("===================================================")
