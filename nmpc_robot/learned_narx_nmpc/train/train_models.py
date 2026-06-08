from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)


def run(cmd):
    print("\n> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    data_files = sorted(Path("data/selected").glob("*.csv"))
    if len(data_files) < 2:
        raise SystemExit(
            "Put at least two good CSV files in train/data/selected/ before training.\n"
            "Keep bad runs in data/raw/ but do not put them in selected/."
        )

    Path("models/narx").mkdir(parents=True, exist_ok=True)

    py = sys.executable

    # This is the model used by run_robot.py.
    run([
        py, "train_narx.py",
        "--runs", "data/selected/*.csv",
        "--state-mode", "bend",
        "--ny", "10",
        "--nu", "10",
        "--ndu", "4",
        "--out-dir", "models/narx",
    ])

    
    run([
        py, "plot_rollouts.py",
        "--model", "models/narx/stable_narx_ridge_bend.npz",
        "--runs", "data/selected/*.csv",
        "--out-dir", "models/narx/rollout_plots",
    ])

    print("\nDone.")
    print("NARX model for the robot: models/narx/stable_narx_ridge_bend.npz")
    print("Now edit params.py if needed, then run: python run_robot.py")


if __name__ == "__main__":
    main()
