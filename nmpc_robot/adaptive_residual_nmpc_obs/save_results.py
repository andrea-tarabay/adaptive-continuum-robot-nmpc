from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_if_columns(df: pd.DataFrame, t, columns: list[str], labels: list[str], path: Path, ylabel: str, title: str | None = None):
    if not all(c in df.columns for c in columns):
        return
    plt.figure(figsize=(10, 5))
    for c, lab in zip(columns, labels):
        plt.plot(t, df[c], label=lab)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_run(rows: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "log.csv"
    df.to_csv(csv_path, index=False)

    if len(df) == 0:
        return csv_path

    t = df["time_s"].to_numpy(float)

    _plot_if_columns(df, t, ["bx_deg", "by_deg", "target_bx_deg", "target_by_deg"], ["bx", "by", "target bx", "target by"], out_dir / "bend_tracking.png", "bend component [deg]", "Bend tracking")
    _plot_if_columns(df, t, ["bend_error_deg"], ["bend error"], out_dir / "bend_error.png", "error [deg]", "Bend error")
    _plot_if_columns(df, t, ["T1_N", "T2_N", "T3_N"], ["T1", "T2", "T3"], out_dir / "tensions.png", "tension [N]", "Commanded tendon tensions")
    _plot_if_columns(df, t, ["dT1_N", "dT2_N", "dT3_N"], ["dT1", "dT2", "dT3"], out_dir / "tension_rate_commands.png", "dT [N/step]", "NMPC tension increments")
    _plot_if_columns(df, t, ["Teff1_N", "Teff2_N", "Teff3_N"], ["Teff1", "Teff2", "Teff3"], out_dir / "effective_tensions.png", "effective tension [N]", "Estimated effective tendon tensions")
    _plot_if_columns(df, t, ["dist_x_rad_s2", "dist_y_rad_s2"], ["dist x", "dist y"], out_dir / "estimated_disturbance.png", "disturbance [rad/s²]", "Online disturbance estimate")
    _plot_if_columns(df, t, ["observer_innovation_deg"], ["observer innovation"], out_dir / "observer_innovation.png", "one-step innovation [deg]", "Observer one-step prediction innovation")
    _plot_if_columns(df, t, ["observer_raw_dist_x_rad_s2", "observer_raw_dist_y_rad_s2"], ["raw dist x", "raw dist y"], out_dir / "observer_raw_disturbance.png", "raw correction [rad/s²]", "Observer raw disturbance correction")
    _plot_if_columns(df, t, ["solve_time_s"], ["solve time"], out_dir / "solve_time.png", "time [s]", "NMPC solve time")

    if all(c in df.columns for c in ["target_bx_deg", "target_by_deg", "bx_deg", "by_deg"]):
        plt.figure(figsize=(7, 7))
        plt.plot(df["target_bx_deg"], df["target_by_deg"], "--", label="target")
        plt.plot(df["bx_deg"], df["by_deg"], label="measured")
        plt.xlabel("bx [deg]")
        plt.ylabel("by [deg]")
        plt.title("Top-down bend-space tracking")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "top_down_bend_space.png", dpi=200)
        plt.close()

    if all(c in df.columns for c in ["phi_deg", "phi_ref_deg", "theta_deg", "theta_ref_deg"]):
        plt.figure(figsize=(10, 6))
        plt.plot(t, df["phi_deg"], label="measured phi")
        plt.plot(t, df["phi_ref_deg"], "--", label="reference phi")
        plt.plot(t, df["theta_deg"], label="measured theta")
        plt.plot(t, df["theta_ref_deg"], "--", label="reference theta")
        plt.xlabel("time [s]")
        plt.ylabel("angle [deg]")
        plt.title("Phi and theta tracking")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "phi_theta_tracking.png", dpi=200)
        plt.close()

    return csv_path


def print_summary(rows: list[dict]):
    if not rows:
        print("No rows logged.")
        return

    df = pd.DataFrame(rows)
    last = df[df["time_s"] >= df["time_s"].max() - 5.0]

    print("\n================ summary, last 5 s ================")
    for cols, label in [
        (["phi_deg", "theta_deg"], "phi/theta [deg]"),
        (["bx_deg", "by_deg"], "bx/by [deg]"),
        (["target_bx_deg", "target_by_deg"], "target bx/by [deg]"),
        (["T1_N", "T2_N", "T3_N"], "tensions [N]"),
        (["dist_x_rad_s2", "dist_y_rad_s2"], "disturbance [rad/s²]"),
        (["observer_innovation_deg"], "observer innovation [deg]"),
    ]:
        if all(c in last.columns for c in cols):
            vals = [round(float(last[c].mean()), 3) for c in cols]
            print(label + ":", *vals)
    if "bend_error_deg" in last.columns:
        print("bend error [deg]:", round(float(last["bend_error_deg"].mean()), 3))
    if "solve_time_s" in last.columns:
        print("solve time mean/max [s]:", round(float(last["solve_time_s"].mean()), 4), round(float(last["solve_time_s"].max()), 4))
    print("===================================================")
