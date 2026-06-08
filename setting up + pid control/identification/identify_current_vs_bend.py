import csv
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from b1_phi_theta import IMUBendReader
from dynamixel_controller import DynamixelController, BaseModel, PortCommError


DXL_PORT = "COM3"
IMU_PORT = "COM4"

MOTOR_IDS = [1]
TEST_MOTOR_ID = 1
DXL_BAUDRATE = 57600
DXL_PROTOCOL = 2.0

REVERSE_DIRECTION = False

BASELINE_CURRENT_MA = {
    1: 0.0,
}

CURRENT_STEPS_MA = [
    0, 20, 40, 60, 80, 100, 120,
    100, 80, 60, 40, 20, 0
]

TEST_CURRENT_SIGN = -1

SETTLE_TIME_S = 1.5
AVERAGE_TIME_S = 0.7
SAMPLE_DT_S = 0.05

PHI_VALID_THRESH_DEG = 3.0
PHI_OFFSET_DEG = 0.0

OUTPUT_ROOT = Path("identification_logs")


def wrap_to_180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles_deg: List[float]) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    s = sum(math.sin(math.radians(a)) for a in angles_deg)
    c = sum(math.cos(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(s, c))


def build_current_command(goal_test_current_mA: float) -> List[float]:
    cmd = []
    for mid in MOTOR_IDS:
        if mid == TEST_MOTOR_ID:
            cmd.append(TEST_CURRENT_SIGN * goal_test_current_mA)
        else:
            cmd.append(BASELINE_CURRENT_MA.get(mid, 0.0))
    return cmd


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

def main():
    if TEST_MOTOR_ID not in MOTOR_IDS:
        raise ValueError("TEST_MOTOR_ID must be included in MOTOR_IDS")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"motor_{TEST_MOTOR_ID}_current_vs_bend_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    timeseries_path = out_dir / "timeseries.csv"
    summary_path = out_dir / "summary.csv"
    notes_path = out_dir / "session_notes.txt"

    imu = None
    dxl = None

    try:
        imu = IMUBendReader(port=IMU_PORT, baudrate=115200, timeout=0.02)

        print("=" * 70)
        print("CURRENT -> BEND IDENTIFICATION")
        print("=" * 70)
        print(f"DXL port     : {DXL_PORT}")
        print(f"DXL baudrate : {DXL_BAUDRATE}")
        print(f"Motor IDs    : {MOTOR_IDS}")
        print(f"Test motor   : {TEST_MOTOR_ID}")
        print()
        print("Hold the arm in the pose you define as STRAIGHT.")
        input("Press Enter to capture the IMU straight reference... ")
        imu.capture_straight_reference(n=60, dt=0.01)
        print("Straight reference captured.\n")

        motor_models = [BaseModel(motor_id=mid) for mid in MOTOR_IDS]
        dxl = DynamixelController(
            port_name=DXL_PORT,
            motor_list=motor_models,
            protocol=DXL_PROTOCOL,
            baudrate=DXL_BAUDRATE,
            reverse_direction=REVERSE_DIRECTION,
        )

        print("Connecting to Dynamixels...")
        dxl.activate_controller()

        dxl.torque_off()
        dxl.set_operating_mode_all("current_control")
        dxl.torque_on()
        dxl.set_goal_current_mA(build_current_command(0.0))
        time.sleep(0.5)

        motor_index = {mid: i for i, mid in enumerate(MOTOR_IDS)}

        # -----------------------------
        # Prepare CSV writers
        # -----------------------------
        timeseries_headers = [
            "wall_time",
            "step_index",
            "goal_current_mA",
            "used_for_average",
            "theta_deg",
            "phi_deg",
            "phi_valid",
            "q_rel_w",
            "q_rel_x",
            "q_rel_y",
            "q_rel_z",
            "test_motor_present_current_mA",
            "test_motor_position_deg",
            "test_motor_velocity_deg_s",
            "test_motor_pwm_percent",
        ]

        for mid in MOTOR_IDS:
            timeseries_headers += [
                f"motor_{mid}_current_mA",
                f"motor_{mid}_position_deg",
                f"motor_{mid}_velocity_deg_s",
                f"motor_{mid}_pwm_percent",
            ]

        summary_headers = [
            "step_index",
            "goal_current_mA",
            "theta_deg_mean",
            "theta_deg_std",
            "phi_deg_circ_mean_valid_only",
            "phi_valid_fraction",
            "test_motor_present_current_mA_mean",
            "test_motor_present_current_mA_std",
            "test_motor_position_deg_mean",
            "test_motor_pwm_percent_mean",
            "n_avg_samples",
        ]

        with open(timeseries_path, "w", newline="") as f_ts, open(summary_path, "w", newline="") as f_sum:
            ts_writer = csv.DictWriter(f_ts, fieldnames=timeseries_headers)
            sum_writer = csv.DictWriter(f_sum, fieldnames=summary_headers)
            ts_writer.writeheader()
            sum_writer.writeheader()

            # -----------------------------
            # Run steps
            # -----------------------------
            for step_index, goal_current_mA in enumerate(CURRENT_STEPS_MA):
                print(f"[Step {step_index + 1}/{len(CURRENT_STEPS_MA)}] goal_current = {goal_current_mA} mA")

                cmd = build_current_command(goal_current_mA)
                dxl.set_goal_current_mA(cmd)

                t0 = time.time()
                avg_rows = []

                while True:
                    elapsed = time.time() - t0
                    if elapsed > (SETTLE_TIME_S + AVERAGE_TIME_S):
                        break

                    used_for_average = int(elapsed >= SETTLE_TIME_S)

                    imu_state = imu.get_state(
                        phi_offset_deg=PHI_OFFSET_DEG,
                        phi_valid_thresh_deg=PHI_VALID_THRESH_DEG,
                    )

                    pos_deg, vel_deg_s, cur_mA, pwm_pct = dxl.read_info_with_unit(
                        pwm_unit="percent",
                        angle_unit="deg",
                        current_unit="mA",
                        fast_read=True,
                    )

                    test_idx = motor_index[TEST_MOTOR_ID]

                    row = {
                        "wall_time": time.time(),
                        "step_index": step_index,
                        "goal_current_mA": goal_current_mA,
                        "used_for_average": used_for_average,
                        "theta_deg": float(imu_state["theta_deg"]),
                        "phi_deg": float(wrap_to_180(float(imu_state["phi_deg"]))),
                        "phi_valid": int(bool(imu_state["phi_valid"])),
                        "q_rel_w": float(imu_state["q_rel_wxyz"][0]),
                        "q_rel_x": float(imu_state["q_rel_wxyz"][1]),
                        "q_rel_y": float(imu_state["q_rel_wxyz"][2]),
                        "q_rel_z": float(imu_state["q_rel_wxyz"][3]),
                        "test_motor_present_current_mA": float(cur_mA[test_idx]),
                        "test_motor_position_deg": float(pos_deg[test_idx]),
                        "test_motor_velocity_deg_s": float(vel_deg_s[test_idx]),
                        "test_motor_pwm_percent": float(pwm_pct[test_idx]),
                    }

                    for mid in MOTOR_IDS:
                        i = motor_index[mid]
                        row[f"motor_{mid}_current_mA"] = float(cur_mA[i])
                        row[f"motor_{mid}_position_deg"] = float(pos_deg[i])
                        row[f"motor_{mid}_velocity_deg_s"] = float(vel_deg_s[i])
                        row[f"motor_{mid}_pwm_percent"] = float(pwm_pct[i])

                    ts_writer.writerow(row)

                    if used_for_average:
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
                    "step_index": step_index,
                    "goal_current_mA": goal_current_mA,
                    "theta_deg_mean": mean_of(avg_rows, "theta_deg"),
                    "theta_deg_std": std_of(avg_rows, "theta_deg"),
                    "phi_deg_circ_mean_valid_only": phi_circ_mean,
                    "phi_valid_fraction": phi_valid_fraction,
                    "test_motor_present_current_mA_mean": mean_of(avg_rows, "test_motor_present_current_mA"),
                    "test_motor_present_current_mA_std": std_of(avg_rows, "test_motor_present_current_mA"),
                    "test_motor_position_deg_mean": mean_of(avg_rows, "test_motor_position_deg"),
                    "test_motor_pwm_percent_mean": mean_of(avg_rows, "test_motor_pwm_percent"),
                    "n_avg_samples": len(avg_rows),
                }
                sum_writer.writerow(summary_row)

                print(
                    f"    theta_mean = {summary_row['theta_deg_mean']:.2f} deg | "
                    f"phi_mean(valid) = {summary_row['phi_deg_circ_mean_valid_only']:.2f} deg | "
                    f"I_present_mean = {summary_row['test_motor_present_current_mA_mean']:.2f} mA"
                )

        dxl.set_goal_current_mA(build_current_command(0.0))
        time.sleep(0.3)
        dxl.torque_off()

        with open(notes_path, "w") as f:
            f.write("Current vs bend identification session\n")
            f.write(f"Timestamp: {stamp}\n")
            f.write(f"DXL_PORT: {DXL_PORT}\n")
            f.write(f"DXL_BAUDRATE: {DXL_BAUDRATE}\n")
            f.write(f"IMU_PORT: {IMU_PORT}\n")
            f.write(f"MOTOR_IDS: {MOTOR_IDS}\n")
            f.write(f"TEST_MOTOR_ID: {TEST_MOTOR_ID}\n")
            f.write(f"CURRENT_STEPS_MA: {CURRENT_STEPS_MA}\n")
            f.write(f"TEST_CURRENT_SIGN: {TEST_CURRENT_SIGN}\n")
            f.write(f"SETTLE_TIME_S: {SETTLE_TIME_S}\n")
            f.write(f"AVERAGE_TIME_S: {AVERAGE_TIME_S}\n")
            f.write(f"SAMPLE_DT_S: {SAMPLE_DT_S}\n")
            f.write(f"PHI_VALID_THRESH_DEG: {PHI_VALID_THRESH_DEG}\n")
            f.write(f"PHI_OFFSET_DEG: {PHI_OFFSET_DEG}\n")

        print("\nDone.")
        print(f"Saved raw data to: {timeseries_path}")
        print(f"Saved summary to:  {summary_path}")
        print(f"Saved notes to:    {notes_path}")

    except PortCommError as e:
        print("\nDynamixel communication failed.")
        print(str(e))
        print("\nCheck these first:")
        print("1. Correct COM port for the Dynamixel adapter")
        print("2. Correct motor ID")
        print("3. External power is on for the motors")
        print("4. Baudrate matches the motor setting")
        print("5. Data cable / power cable are connected properly")
        print("\nStart with MOTOR_IDS = [1] and try DXL_BAUDRATE = 57600 or 1000000.")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        if dxl is not None:
            try:
                dxl.set_goal_current_mA(build_current_command(0.0))
                time.sleep(0.2)
                dxl.torque_off()
            except Exception:
                pass
    finally:
        if imu is not None:
            imu.close()


if __name__ == "__main__":
    main()