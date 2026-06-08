import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================


LOGS = [
    {
        "path": r"C:\Users\Admin\OneDrive\Desktop\EPFL_MSc_Robotics\SPRING_2026\nmpc_final\pcc_physics_nmpc\csv_and_plots_adapt\hardware_tests\hardware_run_20260608_105658\pcc_combined_log.csv",
        "label": "PCC",
        "color": "#E66100",
    },
    {
        "path": r"C:\Users\Admin\OneDrive\Desktop\EPFL_MSc_Robotics\SPRING_2026\nmpc_final\learned_narx_nmpc\hardware_validation\acados_narx_nmpc_circle_20260608_074935\log.csv",
        "label": "Learned\nNARX",
        "color": "#008B8B",
    },
    {
        "path": r"C:\Users\Admin\OneDrive\Desktop\DATA\GRAYBOX\adaptive_residual_graybox_acados_circle_50deg_ru2\log.csv",
        "label": "Adaptive\ngray-box",
        "color": "#8E5EA2",
    },
]
OUTPUT_PDF = "trajectory_tracking_comparison.pdf"
OUTPUT_PNG = "trajectory_tracking_comparison.png"

METRIC_START_TIME_S = 0.0

REAL_TIME_LIMIT_MS = 100.0


# =========================
# HELPER FUNCTIONS
# =========================

def load_log(path):
    df = pd.read_csv(path)

    required_cols = [
        "time_s",
        "bx_deg",
        "by_deg",
        "target_bx_deg",
        "target_by_deg",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    if "bend_error_deg" not in df.columns:
        df["bend_error_deg"] = np.sqrt(
            (df["target_bx_deg"] - df["bx_deg"])**2
            + (df["target_by_deg"] - df["by_deg"])**2
        )

    if "solve_time_s" not in df.columns:
        df["solve_time_s"] = np.nan

    return df


def compute_metrics(df, metric_start_time_s=0.0):
    mask = df["time_s"] >= metric_start_time_s
    d = df.loc[mask].copy()

    err = d["bend_error_deg"].to_numpy()
    solve_ms = d["solve_time_s"].to_numpy() * 1000.0

    err = err[np.isfinite(err)]
    solve_ms = solve_ms[np.isfinite(solve_ms)]

    return {
        "mean_error_deg": np.mean(err),
        "rms_error_deg": np.sqrt(np.mean(err**2)),
        "p95_error_deg": np.percentile(err, 95),
        "max_error_deg": np.max(err),

        "mean_solve_ms": np.mean(solve_ms),
        "p95_solve_ms": np.percentile(solve_ms, 95),
        "max_solve_ms": np.max(solve_ms),
    }


# =========================
# LOAD DATA
# =========================

runs = []

for item in LOGS:
    df = load_log(item["path"])
    metrics = compute_metrics(df, METRIC_START_TIME_S)

    runs.append({
        "df": df,
        "label": item["label"],
        "color": item["color"],
        "metrics": metrics,
    })


# =========================
# PRINT METRICS
# =========================

print("\nTracking and computation metrics")
print("--------------------------------")

for run in runs:
    m = run["metrics"]
    print(run["label"].replace("\n", " "))
    print(f"  Mean error      = {m['mean_error_deg']:.2f} deg")
    print(f"  RMS error       = {m['rms_error_deg']:.2f} deg")
    print(f"  95th error      = {m['p95_error_deg']:.2f} deg")
    print(f"  Max error       = {m['max_error_deg']:.2f} deg")
    print(f"  Mean solve time = {m['mean_solve_ms']:.2f} ms")
    print(f"  95th solve time = {m['p95_solve_ms']:.2f} ms")
    print(f"  Max solve time  = {m['max_solve_ms']:.2f} ms")
    print()

# =========================
# PLOT
# =========================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

ax_path, ax_err, ax_time = axes


# -------------------------
# (a) Bend-space path
# -------------------------

# Use the reference from the first log.
# This is still the actual logged reference trajectory.
ref_df = runs[0]["df"]

ax_path.plot(
    ref_df["target_bx_deg"],
    ref_df["target_by_deg"],
    "k--",
    linewidth=1.6,
    label="Reference",
)

for run in runs:
    df = run["df"]
    ax_path.plot(
        df["bx_deg"],
        df["by_deg"],
        color=run["color"],
        linewidth=1.6,
        label=run["label"].replace("\n", " "),
    )

ax_path.set_title("(a) Bend-space path", fontweight="bold")
ax_path.set_xlabel(r"$b_x$ [deg]")
ax_path.set_ylabel(r"$b_y$ [deg]")
ax_path.axis("equal")
ax_path.grid(True, alpha=0.25)
ax_path.legend(loc="best", frameon=True)


# -------------------------
# (b) Tracking error
# -------------------------

labels = [run["label"] for run in runs]
colors = [run["color"] for run in runs]
mean_errors = [run["metrics"]["mean_error_deg"] for run in runs]

x = np.arange(len(runs))

bars = ax_err.bar(
    x,
    mean_errors,
    color=colors,
    edgecolor="black",
    linewidth=0.9,
    width=0.55,
)

# Add enough space above bars
ax_err.set_ylim(0, max(mean_errors) * 1.25)

# Big readable values inside/above bars
for bar, value in zip(bars, mean_errors):
    ax_err.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(mean_errors) * 0.04,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )

ax_err.set_title("(b) Tracking error", fontweight="bold")
ax_err.set_ylabel("Mean bend error [deg]")
ax_err.set_xticks(x)
ax_err.set_xticklabels(labels)
ax_err.grid(True, axis="y", alpha=0.25)

# -------------------------
# (c) Computation time - mean and 95th percentile
# -------------------------

mean_times = [run["metrics"]["mean_solve_ms"] for run in runs]
p95_times = [run["metrics"]["p95_solve_ms"] for run in runs]
max_times = [run["metrics"]["max_solve_ms"] for run in runs]

bar_width = 0.32
x = np.arange(len(runs))

bars_mean = ax_time.bar(
    x - bar_width / 2,
    mean_times,
    width=bar_width,
    color=colors,
    edgecolor="black",
    linewidth=0.9,
    label="Mean",
)

bars_p95 = ax_time.bar(
    x + bar_width / 2,
    p95_times,
    width=bar_width,
    color=colors,
    edgecolor="black",
    linewidth=0.9,
    alpha=0.45,
    hatch="//",
    label="95th perc.",
)

ax_time.axhline(
    REAL_TIME_LIMIT_MS,
    color="black",
    linestyle="--",
    linewidth=1.0,
    alpha=0.7,
)

ax_time.text(
    0.98,
    REAL_TIME_LIMIT_MS + 3,
    "100 ms",
    transform=ax_time.get_yaxis_transform(),
    ha="right",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)

for bars in [bars_mean, bars_p95]:
    for bar in bars:
        value = bar.get_height()
        ax_time.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(p95_times) * 0.04,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

ax_time.set_ylim(0, max(max(p95_times), REAL_TIME_LIMIT_MS) * 1.25)

ax_time.set_title("(c) Computation time", fontweight="bold")
ax_time.set_ylabel("Solve time [ms]")
ax_time.set_xticks(x)
ax_time.set_xticklabels(labels)
ax_time.grid(True, axis="y", alpha=0.25)
ax_time.legend(frameon=True, fontsize=9)
# =========================
# SAVE
# =========================

fig.tight_layout()

fig.savefig(OUTPUT_PDF, bbox_inches="tight")
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")

plt.show()

print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")