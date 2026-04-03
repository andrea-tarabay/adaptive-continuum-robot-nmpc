from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def circular_mean_deg(angles_deg):
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[~np.isnan(angles_deg)]
    if len(angles_deg) == 0:
        return np.nan
    s = np.sum(np.sin(np.deg2rad(angles_deg)))
    c = np.sum(np.cos(np.deg2rad(angles_deg)))
    return np.rad2deg(np.arctan2(s, c))


def wrap_to_360(angle_deg):
    return angle_deg % 360.0


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for mass_g, g in summary.groupby("mass_g", sort=True):
        theta_vals = g["theta_deg_mean"].astype(float).to_numpy()
        theta_std_vals = g["theta_deg_std"].astype(float).to_numpy()

        phi_vals = pd.to_numeric(
            g["phi_deg_circ_mean_valid_only"], errors="coerce"
        ).to_numpy()

        rows.append(
            {
                "mass_g": float(mass_g),
                "force_N": float(g["force_N"].mean()) if "force_N" in g.columns else np.nan,
                "n_trials": int(len(g)),
                "theta_mean_deg": float(np.mean(theta_vals)),
                "theta_std_between_trials_deg": float(np.std(theta_vals, ddof=0)),
                "theta_std_within_trial_mean_deg": float(np.mean(theta_std_vals)),
                "phi_mean_deg": float(circular_mean_deg(phi_vals)),
            }
        )

    return pd.DataFrame(rows).sort_values("mass_g").reset_index(drop=True)


def plot_summary_only(folder_path: str, show_plots: bool = True):
    folder = Path(folder_path)
    summary_path = folder / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")

    summary = pd.read_csv(summary_path)

    for col in [
        "mass_g",
        "force_N",
        "theta_deg_mean",
        "theta_deg_std",
        "phi_deg_circ_mean_valid_only",
        "phi_valid_fraction",
    ]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce")

    agg = aggregate_summary(summary)
    agg.to_csv(folder / "aggregated_from_summary_only.csv", index=False)

    # --------------------------------------------------
    # Plot 1: raw theta per trial
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(summary["mass_g"], summary["theta_deg_mean"], "o")
    plt.xlabel("Mass [g]")
    plt.ylabel("Theta mean [deg]")
    plt.title("Raw summary points: theta vs mass")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "raw_theta_vs_mass_from_summary.png", dpi=200)

    # --------------------------------------------------
    # Plot 2: aggregated theta vs mass
    # --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.errorbar(
        agg["mass_g"],
        agg["theta_mean_deg"],
        yerr=agg["theta_std_between_trials_deg"],
        marker="o",
        capsize=4,
    )
    plt.xlabel("Mass [g]")
    plt.ylabel("Theta mean [deg]")
    plt.title("Aggregated theta vs mass")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(folder / "aggregated_theta_vs_mass.png", dpi=200)

   # --------------------------------------------------
    # Plot 3: aggregated theta vs force
    # --------------------------------------------------
    if "force_N" in agg.columns and not agg["force_N"].isna().all():
        theta_force = agg.dropna(subset=["force_N", "theta_mean_deg"]).copy()
        theta_force = theta_force.sort_values("force_N")

        plt.figure(figsize=(7, 5))
        plt.errorbar(
            theta_force["force_N"],
            theta_force["theta_mean_deg"],
            yerr=theta_force["theta_std_between_trials_deg"],
            marker="o",
            capsize=4,
            label="Data",
        )

        if len(theta_force) >= 2:
            x = theta_force["force_N"].to_numpy(dtype=float)
            y = theta_force["theta_mean_deg"].to_numpy(dtype=float)

            slope_theta, intercept_theta = np.polyfit(x, y, 1)

            x_fit = np.linspace(x.min(), x.max(), 200)
            y_fit = slope_theta * x_fit + intercept_theta

            plt.plot(
                x_fit,
                y_fit,
                "-",
                label=f"Fit: slope = {slope_theta:.3f} deg/N",
            )

            print(f"Theta vs force slope = {slope_theta:.6f} deg/N")

        plt.xlabel("Force [N]")
        plt.ylabel("Theta mean [deg]")
        plt.title("Aggregated theta vs force")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(folder / "aggregated_theta_vs_force.png", dpi=200)
      # --------------------------------------------------
    # Plot 4: phi vs mass
    # --------------------------------------------------
    phi_plot = agg.dropna(subset=["phi_mean_deg"]).copy()
    if len(phi_plot) > 0:
        phi_plot["phi_wrapped_360"] = phi_plot["phi_mean_deg"].apply(wrap_to_360)

        plt.figure(figsize=(7, 5))
        plt.plot(
            phi_plot["mass_g"],
            phi_plot["phi_wrapped_360"],
            marker="o",
        )
        plt.xlabel("Mass [g]")
        plt.ylabel("Phi [deg]")
        plt.title("Aggregated phi vs mass")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(folder / "aggregated_phi_vs_mass.png", dpi=200)

        # --------------------------------------------------
        # Plot 4b: phi vs force + slope
        # --------------------------------------------------
        if "force_N" in phi_plot.columns and not phi_plot["force_N"].isna().all():
            phi_force = phi_plot.dropna(subset=["force_N"]).copy()
            phi_force = phi_force.sort_values("force_N")

            if len(phi_force) >= 2:
                x = phi_force["force_N"].to_numpy(dtype=float)

                phi_unwrapped = np.rad2deg(
                    np.unwrap(np.deg2rad(phi_force["phi_wrapped_360"].to_numpy(dtype=float)))
                )

                slope, intercept = np.polyfit(x, phi_unwrapped, 1)

                x_fit = np.linspace(x.min(), x.max(), 200)
                y_fit = slope * x_fit + intercept

                plt.figure(figsize=(7, 5))
                plt.plot(x, phi_unwrapped, "o", label="Data")
                plt.plot(x_fit, y_fit, "-", label=f"Fit: slope = {slope:.3f} deg/N")
                plt.xlabel("Force [N]")
                plt.ylabel("Phi [deg]")
                plt.title("Aggregated phi vs force")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(folder / "aggregated_phi_vs_force_with_slope.png", dpi=200)

                print(f"Phi vs force slope = {slope:.6f} deg/N")

        # Polar plot
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="polar")
        ax.scatter(
            np.deg2rad(phi_plot["phi_wrapped_360"]),
            phi_plot["theta_mean_deg"],
            s=70,
        )

        for _, row in phi_plot.iterrows():
            ax.text(
                math.radians(row["phi_wrapped_360"]),
                row["theta_mean_deg"],
                f"{row['mass_g']:.0f} g",
            )

        ax.set_title("Polar bend map from summary only")
        plt.tight_layout()
        plt.savefig(folder / "aggregated_phi_theta_polar.png", dpi=200)
    # --------------------------------------------------
    # Plot 5: phi-valid fraction
    # --------------------------------------------------
    if "phi_valid_fraction" in summary.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(summary["mass_g"], summary["phi_valid_fraction"], "o-")
        plt.xlabel("Mass [g]")
        plt.ylabel("phi_valid fraction")
        plt.title("Phi validity from summary only")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(folder / "phi_valid_fraction_from_summary.png", dpi=200)

    print("\nSaved plots in:")
    print(folder)

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    plot_summary_only(
        r"C:\Users\Poseidon\Desktop\dxl_test\mass_bend_logs_interactive\interactive_mass_vs_bend_20260317_224814",
        show_plots=False
    )