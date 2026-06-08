import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

from b1_phi_theta import IMUBendReader


IMU_PORT = "COM4"
IMU_BAUDRATE = 115200

OUTPUT_ROOT = Path("mass_bend_logs_interactive")

PHI_VALID_THRESH_DEG = 3.0
PHI_OFFSET_DEG = 0.0

STRAIGHT_CAPTURE_SAMPLES = 60
STRAIGHT_CAPTURE_DT_S = 0.01

AVERAGE_TIME_S = 2.0
SAMPLE_DT_S = 0.05

G = 9.81


def wrap_to_180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles_deg: List[float]) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    s = sum(math.sin(math.radians(a)) for a in angles_deg)
    c = sum(math.cos(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(s, c))


def mean_of(rows: List[Dict], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def std_of(rows: List[Dict], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    if len(vals) == 0:
        return float("nan")
    return float(np.std(vals))


def mass_g_to_force_N(mass_g: float) -> float:
    return (mass_g / 1000.0) * G


def ask_float(prompt: str) -> float:
    while True:
        s = input(prompt).strip()
        try:
            return float(s)
        except ValueError:
            print("Please enter a valid number.")


def ask_int(prompt: str) -> int:
    while True:
        s = input(prompt).strip()
        try:
            return int(s)
        except ValueError:
            print("Please enter a valid integer.")


def ask_yes_no(prompt: str) -> bool:
    while True:
        s = input(prompt).strip().lower()
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please answer with y/n.")


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"interactive_mass_vs_bend_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = out_dir / "timeseries.csv"
    summary_path = out_dir / "summary.csv"
    notes_path = out_dir / "session_notes.txt"

    imu = None

    try:
        imu = IMUBendReader(
            port=IMU_PORT,
            baudrate=IMU_BAUDRATE,
            timeout=0.02,
        )

        print("=" * 72)
        print("INTERACTIVE HANGING MASS -> BEND LOGGER")
        print("=" * 72)
        print("For every trial:")
        print("  1) straighten the arm again")
        print("  2) capture new straight reference")
        print("  3) hang the mass")
        print("  4) type the mass")
        print("  5) press Enter when stable")
        print("  6) script logs IMU data")
        print()

        tendon_id = ask_int("Which tendon are you testing in this session? ")

        timeseries_headers = [
            "wall_time",
            "trial_index",
            "tendon_id",
            "mass_g",
            "force_N",
            "used_for_average",
            "theta_deg",
            "phi_deg",
            "phi_valid",
            "q_rel_w",
            "q_rel_x",
            "q_rel_y",
            "q_rel_z",
        ]

        summary_headers = [
            "trial_index",
            "tendon_id",
            "mass_g",
            "force_N",
            "theta_deg_mean",
            "theta_deg_std",
            "phi_deg_circ_mean_valid_only",
            "phi_valid_fraction",
            "n_avg_samples",
        ]

        with open(timeseries_path, "w", newline="") as f_ts, open(summary_path, "w", newline="") as f_sum:
            ts_writer = csv.DictWriter(f_ts, fieldnames=timeseries_headers)
            sum_writer = csv.DictWriter(f_sum, fieldnames=summary_headers)
            ts_writer.writeheader()
            sum_writer.writeheader()

            trial_index = 0

            while True:
                print("\n" + "-" * 72)
                print(f"TRIAL {trial_index + 1}")

                print("Straighten the arm now.")
                input("Press Enter to capture a NEW straight reference... ")

                imu.capture_straight_reference(
                    n=STRAIGHT_CAPTURE_SAMPLES,
                    dt=STRAIGHT_CAPTURE_DT_S,
                )
                print("Straight reference updated for this trial.")

                print("\nNow hang the mass.")
                mass_g = ask_float("Type the mass you hung [g]: ")
                force_N = mass_g_to_force_N(mass_g)

                input("When the arm looks stable, press Enter to start logging... ")

                print(f"Logging for {AVERAGE_TIME_S:.1f} s...")
                t0 = time.time()
                avg_rows = []

                while True:
                    elapsed = time.time() - t0
                    if elapsed > AVERAGE_TIME_S:
                        break

                    imu_state = imu.get_state(
                        phi_offset_deg=PHI_OFFSET_DEG,
                        phi_valid_thresh_deg=PHI_VALID_THRESH_DEG,
                    )

                    row = {
                        "wall_time": time.time(),
                        "trial_index": trial_index,
                        "tendon_id": tendon_id,
                        "mass_g": float(mass_g),
                        "force_N": float(force_N),
                        "used_for_average": 1,
                        "theta_deg": float(imu_state["theta_deg"]),
                        "phi_deg": float(wrap_to_180(float(imu_state["phi_deg"]))),
                        "phi_valid": int(bool(imu_state["phi_valid"])),
                        "q_rel_w": float(imu_state["q_rel_wxyz"][0]),
                        "q_rel_x": float(imu_state["q_rel_wxyz"][1]),
                        "q_rel_y": float(imu_state["q_rel_wxyz"][2]),
                        "q_rel_z": float(imu_state["q_rel_wxyz"][3]),
                    }

                    ts_writer.writerow(row)
                    avg_rows.append(row)

                    time.sleep(SAMPLE_DT_S)

                valid_phi_rows = [r for r in avg_rows if int(r["phi_valid"]) == 1]
                phi_valid_fraction = (
                    float(len(valid_phi_rows)) / float(len(avg_rows))
                    if len(avg_rows) > 0 else float("nan")
                )
                phi_circ_mean = (
                    circular_mean_deg([float(r["phi_deg"]) for r in valid_phi_rows])
                    if len(valid_phi_rows) > 0 else float("nan")
                )

                summary_row = {
                    "trial_index": trial_index,
                    "tendon_id": tendon_id,
                    "mass_g": float(mass_g),
                    "force_N": float(force_N),
                    "theta_deg_mean": mean_of(avg_rows, "theta_deg"),
                    "theta_deg_std": std_of(avg_rows, "theta_deg"),
                    "phi_deg_circ_mean_valid_only": phi_circ_mean,
                    "phi_valid_fraction": phi_valid_fraction,
                    "n_avg_samples": len(avg_rows),
                }
                sum_writer.writerow(summary_row)

                print("\nTrial saved:")
                print(f"  mass = {mass_g:.3f} g")
                print(f"  force = {force_N:.5f} N")
                print(f"  theta_mean = {summary_row['theta_deg_mean']:.3f} deg")
                print(f"  theta_std  = {summary_row['theta_deg_std']:.3f} deg")
                print(f"  phi_mean(valid) = {summary_row['phi_deg_circ_mean_valid_only']:.3f} deg")
                print(f"  phi_valid_fraction = {summary_row['phi_valid_fraction']:.3f}")

                trial_index += 1

                again = ask_yes_no("\nDo another trial? [y/n]: ")
                if not again:
                    break

        with open(notes_path, "w") as f:
            f.write("Interactive hanging mass vs bend logger\n")
            f.write(f"Timestamp: {stamp}\n")
            f.write(f"IMU_PORT: {IMU_PORT}\n")
            f.write(f"TENDON_ID: {tendon_id}\n")
            f.write(f"STRAIGHT_CAPTURE_SAMPLES: {STRAIGHT_CAPTURE_SAMPLES}\n")
            f.write(f"STRAIGHT_CAPTURE_DT_S: {STRAIGHT_CAPTURE_DT_S}\n")
            f.write(f"AVERAGE_TIME_S: {AVERAGE_TIME_S}\n")
            f.write(f"SAMPLE_DT_S: {SAMPLE_DT_S}\n")
            f.write(f"PHI_VALID_THRESH_DEG: {PHI_VALID_THRESH_DEG}\n")
            f.write(f"PHI_OFFSET_DEG: {PHI_OFFSET_DEG}\n")

        print("\nDone.")
        print(f"Saved raw data to: {timeseries_path}")
        print(f"Saved summary to:  {summary_path}")
        print(f"Saved notes to:    {notes_path}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if imu is not None:
            imu.close()


if __name__ == "__main__":
    main()