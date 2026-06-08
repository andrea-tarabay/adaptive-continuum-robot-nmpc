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

ALL_MOTOR_IDS = [1, 2, 3]
TEST_MOTOR_IDS = [1, 2, 3]
N_REPEATS_PER_MOTOR = 3

DXL_BAUDRATE = 57600
DXL_PROTOCOL = 2.0
REVERSE_DIRECTION = False

BASELINE_CURRENT_MA = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
}

CURRENT_SIGN_BY_MOTOR = {
    1: -1,
    2: -1,
    3: -1,
}

CURRENT_STEPS_MA = [
    0, 20, 40, 60, 80, 100, 120,
    100, 80, 60, 40, 20, 0
]

SETTLE_TIME_S = 1.5
AVERAGE_TIME_S = 0.7
SAMPLE_DT_S = 0.05

PHI_VALID_THRESH_DEG = 3.0
PHI_OFFSET_DEG = 0.0

ZERO_CURRENT_WAIT_S = 0.5
BETWEEN_RUN_WAIT_S = 0.5

OUTPUT_ROOT = Path("identification_logs")


# Helpers
def wrap_to_180(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles_deg: List[float]) -> float:
    if len(angles_deg) == 0:
        return float("nan")
    s = sum(math.sin(math.radians(a)) for a in angles_deg)
    c = sum(math.cos(math.radians(a)) for a in angles_deg)
    return math.degrees(math.atan2(s, c))


def build_current_command(test_motor_id: int, goal_test_current_mA: float) -> List[float]:
    cmd = []
    sign = CURRENT_SIGN_BY_MOTOR[test_motor_id]
    for mid in ALL_MOTOR_IDS:
        if mid == test_motor_id:
            cmd.append(sign * goal_test_current_mA)
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


def zero_all_currents(dxl, wait_s: float = 0.3):
    cmd = [0.0 for _ in ALL_MOTOR_IDS]
    dxl.set_goal_current_mA(cmd)
    time.sleep(wait_s)


def run_single_identification(
    imu: IMUBendReader,
    dxl: DynamixelController,
    motor_index: Dict[int, int],
    test_motor_id: int,
    repeat_idx: int,
    out_dir: Path,
    aggregate_summary_writer=None,
):
    timeseries_path = out_dir / "timeseries.csv"
    summary_path = out_dir / "summary.csv"
    notes_path = out_dir / "session_notes.txt"

    print("=" * 70)
    print(f"Motor {test_motor_id} | Repeat {repeat_idx}")
    print("=" * 70)
    print("Please straighten the arm now.")
    input("Press Enter when the arm is straight and ready for reference capture... ")

    imu.capture_straight_reference(n=60, dt=0.01)
    print("Straight reference captured.\n")

    zero_all_currents(dxl, wait_s=ZERO_CURRENT_WAIT_S)

    timeseries_headers = [
        "run_timestamp",
        "test_motor_id",
        "repeat_idx",
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

    for mid in ALL_MOTOR_IDS:
        timeseries_headers += [
            f"motor_{mid}_current_mA",
            f"motor_{mid}_position_deg",
            f"motor_{mid}_velocity_deg_s",
            f"motor_{mid}_pwm_percent",
        ]

    summary_headers = [
        "run_timestamp",
        "test_motor_id",
        "repeat_idx",
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

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(timeseries_path, "w", newline="") as f_ts, open(summary_path, "w", newline="") as f_sum:
        ts_writer = csv.DictWriter(f_ts, fieldnames=timeseries_headers)
        sum_writer = csv.DictWriter(f_sum, fieldnames=summary_headers)
        ts_writer.writeheader()
        sum_writer.writeheader()

        for step_index, goal_current_mA in enumerate(CURRENT_STEPS_MA):
            print(
                f"[Motor {test_motor_id} | Repeat {repeat_idx} | "
                f"Step {step_index + 1}/{len(CURRENT_STEPS_MA)}] "
                f"goal_current = {goal_current_mA} mA"
            )

            cmd = build_current_command(test_motor_id, goal_current_mA)
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

                test_idx = motor_index[test_motor_id]

                row = {
                    "run_timestamp": run_stamp,
                    "test_motor_id": test_motor_id,
                    "repeat_idx": repeat_idx,
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

                for mid in ALL_MOTOR_IDS:
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
                "run_timestamp": run_stamp,
                "test_motor_id": test_motor_id,
                "repeat_idx": repeat_idx,
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

            if aggregate_summary_writer is not None:
                aggregate_summary_writer.writerow(summary_row)

            print(
                f"    theta_mean = {summary_row['theta_deg_mean']:.2f} deg | "
                f"phi_mean(valid) = {summary_row['phi_deg_circ_mean_valid_only']:.2f} deg | "
                f"I_present_mean = {summary_row['test_motor_present_current_mA_mean']:.2f} mA"
            )

    zero_all_currents(dxl, wait_s=ZERO_CURRENT_WAIT_S)

    with open(notes_path, "w") as f:
        f.write("Current vs bend identification session\n")
        f.write(f"run_timestamp: {run_stamp}\n")
        f.write(f"test_motor_id: {test_motor_id}\n")
        f.write(f"repeat_idx: {repeat_idx}\n")
        f.write(f"DXL_PORT: {DXL_PORT}\n")
        f.write(f"DXL_BAUDRATE: {DXL_BAUDRATE}\n")
        f.write(f"IMU_PORT: {IMU_PORT}\n")
        f.write(f"ALL_MOTOR_IDS: {ALL_MOTOR_IDS}\n")
        f.write(f"CURRENT_STEPS_MA: {CURRENT_STEPS_MA}\n")
        f.write(f"CURRENT_SIGN_BY_MOTOR: {CURRENT_SIGN_BY_MOTOR}\n")
        f.write(f"SETTLE_TIME_S: {SETTLE_TIME_S}\n")
        f.write(f"AVERAGE_TIME_S: {AVERAGE_TIME_S}\n")
        f.write(f"SAMPLE_DT_S: {SAMPLE_DT_S}\n")
        f.write(f"PHI_VALID_THRESH_DEG: {PHI_VALID_THRESH_DEG}\n")
        f.write(f"PHI_OFFSET_DEG: {PHI_OFFSET_DEG}\n")

    print(f"\nSaved run data to: {out_dir}\n")


def main():
    for mid in TEST_MOTOR_IDS:
        if mid not in ALL_MOTOR_IDS:
            raise ValueError(f"Test motor {mid} is not included in ALL_MOTOR_IDS")

    session_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_ROOT / f"multi_motor_identification_{session_stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    aggregate_summary_path = session_dir / "aggregate_summary.csv"
    session_notes_path = session_dir / "session_notes.txt"

    imu = None
    dxl = None

    aggregate_summary_headers = [
        "run_timestamp",
        "test_motor_id",
        "repeat_idx",
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

    try:
        imu = IMUBendReader(port=IMU_PORT, baudrate=115200, timeout=0.02)

        motor_models = [BaseModel(motor_id=mid) for mid in ALL_MOTOR_IDS]
        dxl = DynamixelController(
            port_name=DXL_PORT,
            motor_list=motor_models,
            protocol=DXL_PROTOCOL,
            baudrate=DXL_BAUDRATE,
            reverse_direction=REVERSE_DIRECTION,
        )

        print("=" * 70)
        print("MULTI-MOTOR CURRENT -> BEND IDENTIFICATION")
        print("=" * 70)
        print(f"DXL port     : {DXL_PORT}")
        print(f"IMU port     : {IMU_PORT}")
        print(f"All motors   : {ALL_MOTOR_IDS}")
        print(f"Test motors  : {TEST_MOTOR_IDS}")
        print(f"Repeats each : {N_REPEATS_PER_MOTOR}")
        print(f"Session dir  : {session_dir}")
        print()

        print("Connecting to Dynamixels...")
        dxl.activate_controller()

        dxl.torque_off()
        dxl.set_operating_mode_all("current_control")
        dxl.torque_on()
        zero_all_currents(dxl, wait_s=ZERO_CURRENT_WAIT_S)

        motor_index = {mid: i for i, mid in enumerate(ALL_MOTOR_IDS)}

        with open(aggregate_summary_path, "w", newline="") as f_agg:
            agg_writer = csv.DictWriter(f_agg, fieldnames=aggregate_summary_headers)
            agg_writer.writeheader()

            for test_motor_id in TEST_MOTOR_IDS:
                for repeat_idx in range(1, N_REPEATS_PER_MOTOR + 1):
                    run_dir = session_dir / f"motor_{test_motor_id}" / f"repeat_{repeat_idx:02d}"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    run_single_identification(
                        imu=imu,
                        dxl=dxl,
                        motor_index=motor_index,
                        test_motor_id=test_motor_id,
                        repeat_idx=repeat_idx,
                        out_dir=run_dir,
                        aggregate_summary_writer=agg_writer,
                    )

                    zero_all_currents(dxl, wait_s=ZERO_CURRENT_WAIT_S)

                    if not (test_motor_id == TEST_MOTOR_IDS[-1] and repeat_idx == N_REPEATS_PER_MOTOR):
                        print("Please manually re-straighten / reset the arm before the next run.")
                        time.sleep(BETWEEN_RUN_WAIT_S)

        zero_all_currents(dxl, wait_s=ZERO_CURRENT_WAIT_S)
        dxl.torque_off()

        with open(session_notes_path, "w") as f:
            f.write("Multi-motor identification session\n")
            f.write(f"session_timestamp: {session_stamp}\n")
            f.write(f"DXL_PORT: {DXL_PORT}\n")
            f.write(f"IMU_PORT: {IMU_PORT}\n")
            f.write(f"ALL_MOTOR_IDS: {ALL_MOTOR_IDS}\n")
            f.write(f"TEST_MOTOR_IDS: {TEST_MOTOR_IDS}\n")
            f.write(f"N_REPEATS_PER_MOTOR: {N_REPEATS_PER_MOTOR}\n")
            f.write(f"DXL_BAUDRATE: {DXL_BAUDRATE}\n")
            f.write(f"CURRENT_STEPS_MA: {CURRENT_STEPS_MA}\n")
            f.write(f"CURRENT_SIGN_BY_MOTOR: {CURRENT_SIGN_BY_MOTOR}\n")
            f.write(f"SETTLE_TIME_S: {SETTLE_TIME_S}\n")
            f.write(f"AVERAGE_TIME_S: {AVERAGE_TIME_S}\n")
            f.write(f"SAMPLE_DT_S: {SAMPLE_DT_S}\n")
            f.write(f"PHI_VALID_THRESH_DEG: {PHI_VALID_THRESH_DEG}\n")
            f.write(f"PHI_OFFSET_DEG: {PHI_OFFSET_DEG}\n")
            f.write(f"aggregate_summary: {aggregate_summary_path}\n")

        print("\nAll runs completed.")
        print(f"Session folder: {session_dir}")
        print(f"Aggregate summary: {aggregate_summary_path}")

    except PortCommError as e:
        print("\nDynamixel communication failed.")
        print(str(e))
        print("\nCheck:")
        print("1. Correct COM port")
        print("2. Correct IDs")
        print("3. Motor power is on")
        print("4. Baudrate matches")
        print("5. Cabling is correct")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        if dxl is not None:
            try:
                zero_all_currents(dxl, wait_s=0.2)
                dxl.torque_off()
            except Exception:
                pass
    finally:
        if imu is not None:
            imu.close()


if __name__ == "__main__":
    main()