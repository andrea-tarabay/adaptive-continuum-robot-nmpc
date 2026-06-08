from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_theta_current_relation(folder_path: str):
    folder = Path(folder_path)
    summary_path = folder / "summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find {summary_path}")

    df = pd.read_csv(summary_path)

    peak_idx = df["goal_current_mA"].abs().idxmax()
    df_up = df.iloc[:peak_idx + 1].copy()
    df_up["current_mA"] = df_up["goal_current_mA"].abs()

    plt.figure(figsize=(7, 5))
    plt.plot(df_up["current_mA"], df_up["theta_deg_mean"], marker="o", linewidth=2)

    plt.xlabel("Current magnitude [mA]")
    plt.ylabel("Theta mean [deg]")
    plt.title("Theta vs current (increasing sweep only)")
    plt.grid(True)
    plt.tight_layout()

    out_path = folder / "plot_theta_vs_current_relation.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_theta_current_relation(
        r"C:\Users\Poseidon\Desktop\dxl_test\identification_logs\motor_1_current_vs_bend_20260317_093643"
    )