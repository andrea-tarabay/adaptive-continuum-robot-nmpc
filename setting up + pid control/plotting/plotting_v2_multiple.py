from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SESSION_FOLDER = Path(
    r"C:\Users\Poseidon\Desktop\dxl_test\identification_logs\multi_motor_identification_20260317_120207"
)

DIRECTION_MAP_CURRENT_MA = None


# =========================================================
# Helpers
# =========================================================
def circular_mean_deg(values_deg):
    vals = np.asarray(values_deg, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan
    s = np.sin(np.deg2rad(vals)).mean()
    c = np.cos(np.deg2rad(vals)).mean()
    return np.rad2deg(np.arctan2(s, c))


def wrap_deg_center(angle_deg, center_deg=180.0):
    a = np.asarray(angle_deg, dtype=float)
    return ((a - center_deg + 180.0) % 360.0) - 180.0 + center_deg


def get_increasing_sweep_only(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (motor, repeat), keep only the increasing-current part:
    from step 0 up to the first maximum-current step.
    """
    out = []

    for (motor_id, repeat_idx), g in df.groupby(["test_motor_id", "repeat_idx"]):
        g = g.sort_values("step_index").copy()
        g["goal_current_abs_mA"] = g["goal_current_mA"].abs()

        peak_pos = int(np.argmax(g["goal_current_abs_mA"].to_numpy()))
        g_up = g.iloc[: peak_pos + 1].copy()
        out.append(g_up)

    if len(out) == 0:
        return pd.DataFrame()

    return pd.concat(out, ignore_index=True)


def aggregate_per_motor(df_up: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate repeats for each motor/current pair.
    """
    rows = []

    for (motor_id, current_mA), g in df_up.groupby(["test_motor_id", "goal_current_abs_mA"]):
        row = {
            "test_motor_id": int(motor_id),
            "current_mA": float(current_mA),
            "theta_mean_deg": float(g["theta_deg_mean"].mean()),
            "theta_std_deg": float(g["theta_deg_mean"].std(ddof=0)),
            "present_current_mean_mA": float(g["test_motor_present_current_mA_mean"].abs().mean()),
            "present_current_std_mA": float(g["test_motor_present_current_mA_mean"].abs().std(ddof=0)),
            "phi_mean_deg": float(circular_mean_deg(g["phi_deg_circ_mean_valid_only"])),
            "phi_valid_fraction_mean": float(g["phi_valid_fraction"].mean()),
            "n_runs": int(len(g)),
        }
        rows.append(row)

    agg = pd.DataFrame(rows).sort_values(["test_motor_id", "current_mA"]).reset_index(drop=True)
    if len(agg) > 0:
        agg["phi_plot_deg"] = wrap_deg_center(agg["phi_mean_deg"], center_deg=180.0)
    return agg


def aggregate_pooled(df_up: pd.DataFrame) -> pd.DataFrame:
    """
    Pool all motors together and treat them as one generalized actuator.
    """
    rows = []

    for current_mA, g in df_up.groupby("goal_current_abs_mA"):
        row = {
            "current_mA": float(current_mA),
            "theta_mean_deg": float(g["theta_deg_mean"].mean()),
            "theta_std_deg": float(g["theta_deg_mean"].std(ddof=0)),
            "present_current_mean_mA": float(g["test_motor_present_current_mA_mean"].abs().mean()),
            "present_current_std_mA": float(g["test_motor_present_current_mA_mean"].abs().std(ddof=0)),
            "phi_mean_deg": float(circular_mean_deg(g["phi_deg_circ_mean_valid_only"])),
            "phi_valid_fraction_mean": float(g["phi_valid_fraction"].mean()),
            "n_samples": int(len(g)),
        }
        rows.append(row)

    pooled = pd.DataFrame(rows).sort_values("current_mA").reset_index(drop=True)
    if len(pooled) > 0:
        pooled["phi_plot_deg"] = wrap_deg_center(pooled["phi_mean_deg"], center_deg=180.0)
    return pooled


def save_plot(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


# =========================================================
# Main plotting
# =========================================================
def main():
    aggregate_summary_path = SESSION_FOLDER / "aggregate_summary.csv"
    if not aggregate_summary_path.exists():
        raise FileNotFoundError(f"Could not find {aggregate_summary_path}")

    df = pd.read_csv(aggregate_summary_path)

    if len(df) == 0:
        raise RuntimeError("aggregate_summary.csv is empty")

    df_up = get_increasing_sweep_only(df)
    if len(df_up) == 0:
        raise RuntimeError("No increasing-sweep data found")

    per_motor = aggregate_per_motor(df_up)
    pooled = aggregate_pooled(df_up)

    plots_dir = SESSION_FOLDER / "aggregated_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated CSVs too
    per_motor_csv = plots_dir / "aggregated_per_motor.csv"
    pooled_csv = plots_dir / "aggregated_pooled_all_motors.csv"
    per_motor.to_csv(per_motor_csv, index=False)
    pooled.to_csv(pooled_csv, index=False)
    print(f"Saved: {per_motor_csv}")
    print(f"Saved: {pooled_csv}")

    motor_ids = sorted(per_motor["test_motor_id"].unique())

    # -----------------------------------------------------
    # Plot 1: one identification curve per motor
    # -----------------------------------------------------
    n_motors = len(motor_ids)
    fig1, axes = plt.subplots(
        nrows=n_motors,
        ncols=1,
        figsize=(8, 4 * n_motors),
        squeeze=False
    )

    for ax, motor_id in zip(axes[:, 0], motor_ids):
        g = per_motor[per_motor["test_motor_id"] == motor_id].sort_values("current_mA")

        ax.errorbar(
            g["current_mA"],
            g["theta_mean_deg"],
            yerr=g["theta_std_deg"],
            marker="o",
            capsize=4,
        )
        ax.set_title(f"Motor {motor_id}: theta vs current")
        ax.set_xlabel("Current magnitude [mA]")
        ax.set_ylabel("Theta [deg]")
        ax.grid(True)

    save_plot(fig1, plots_dir / "plot_identification_per_motor.png")

    # -----------------------------------------------------
    # Plot 2: all motors overlaid
    # -----------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for motor_id in motor_ids:
        g = per_motor[per_motor["test_motor_id"] == motor_id].sort_values("current_mA")
        ax2.errorbar(
            g["current_mA"],
            g["theta_mean_deg"],
            yerr=g["theta_std_deg"],
            marker="o",
            capsize=3,
            label=f"Motor {motor_id}",
        )

    ax2.set_title("All motors: theta vs current")
    ax2.set_xlabel("Current magnitude [mA]")
    ax2.set_ylabel("Theta [deg]")
    ax2.grid(True)
    ax2.legend()
    save_plot(fig2, plots_dir / "plot_identification_all_motors_overlay.png")

    # -----------------------------------------------------
    # Plot 3: generalized pooled actuator curve
    # -----------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(8, 5))

    ax3.errorbar(
        pooled["current_mA"],
        pooled["theta_mean_deg"],
        yerr=pooled["theta_std_deg"],
        marker="o",
        capsize=4,
    )

    ax3.set_title("Generalized pooled relation: theta vs current")
    ax3.set_xlabel("Current magnitude [mA]")
    ax3.set_ylabel("Theta [deg]")
    ax3.grid(True)
    save_plot(fig3, plots_dir / "plot_identification_pooled_generalized.png")

    # -----------------------------------------------------
    # Plot 4: phi vs current for each motor
    # -----------------------------------------------------
    fig4, ax4 = plt.subplots(figsize=(8, 5))

    for motor_id in motor_ids:
        g = per_motor[per_motor["test_motor_id"] == motor_id].sort_values("current_mA")
        ax4.plot(
            g["current_mA"],
            g["phi_plot_deg"],
            marker="o",
            label=f"Motor {motor_id}",
        )

    ax4.set_title("Bend direction phi vs current")
    ax4.set_xlabel("Current magnitude [mA]")
    ax4.set_ylabel("Phi [deg] (wrapped around 180)")
    ax4.grid(True)
    ax4.legend()
    save_plot(fig4, plots_dir / "plot_phi_vs_current_per_motor.png")

    # -----------------------------------------------------
    # Plot 5: generalized motor direction map
    # angle = phi direction
    # radius = theta magnitude
    # -----------------------------------------------------
    if DIRECTION_MAP_CURRENT_MA is None:
        direction_current = float(per_motor["current_mA"].max())
    else:
        direction_current = float(DIRECTION_MAP_CURRENT_MA)

    direction_rows = []
    for motor_id in motor_ids:
        g = per_motor[per_motor["test_motor_id"] == motor_id].copy()
        idx = (g["current_mA"] - direction_current).abs().idxmin()
        direction_rows.append(g.loc[idx])

    direction_df = pd.DataFrame(direction_rows).reset_index(drop=True)

    fig5 = plt.figure(figsize=(7, 7))
    ax5 = fig5.add_subplot(111, projection="polar")

    theta_rad = np.deg2rad(direction_df["phi_mean_deg"].to_numpy())
    r_vals = direction_df["theta_mean_deg"].to_numpy()

    ax5.scatter(theta_rad, r_vals, s=80)

    for _, row in direction_df.iterrows():
        ax5.text(
            math.radians(float(row["phi_mean_deg"])),
            float(row["theta_mean_deg"]) + 1.0,
            f"M{int(row['test_motor_id'])}",
            ha="center",
            va="center",
        )

    ax5.set_title(
        f"Motor bend-direction map at ~{direction_current:.0f} mA\n"
        f"(angle = phi, radius = theta)"
    )
    save_plot(fig5, plots_dir / "plot_motor_direction_map_polar.png")

    # -----------------------------------------------------
    # Plot 6: motor number vs bend direction at high current
    # -----------------------------------------------------
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    ax6.plot(
        direction_df["test_motor_id"],
        wrap_deg_center(direction_df["phi_mean_deg"], center_deg=180.0),
        marker="o",
    )
    ax6.set_title(f"Motor number vs bend direction at ~{direction_current:.0f} mA")
    ax6.set_xlabel("Motor ID")
    ax6.set_ylabel("Phi [deg] (wrapped around 180)")
    ax6.grid(True)
    save_plot(fig6, plots_dir / "plot_motor_number_vs_phi.png")

    plt.show()


if __name__ == "__main__":
    main()