import traceback
import numpy as np
from tqdm import tqdm
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from acados_utils import setup_ocp_solver, mpc_step_acados
from pcc_arm import PCCSoftArm
from parameters_friction import (
    ARM_PARAMETERS,
    MPC_PARAMETERS,
    RUN_PARAMETERS,
    HARDWARE_PARAMETERS,
    ACADOS_PARAMETERS,
)
from visualisation_v2 import history_plot
from hardware_robot_friction import RobotHardware


# ============================================================
# Hardware config
# ============================================================

imu_port = HARDWARE_PARAMETERS["imu_port"]
dxl_port = HARDWARE_PARAMETERS["dxl_port"]

motor_ids = HARDWARE_PARAMETERS["motor_ids"]

gain_ma_per_n = HARDWARE_PARAMETERS["gain_ma_per_n"]
offset_ma = HARDWARE_PARAMETERS["offset_ma"]
tighten_sign = HARDWARE_PARAMETERS["tighten_sign"]
i_max_ma = HARDWARE_PARAMETERS["i_max_ma"]
phi_offset_deg = HARDWARE_PARAMETERS["phi_offset_deg"]

# ============================================================
# Debug print config
# ============================================================

DEBUG_PRINT = True
DEBUG_EVERY = 5

# ============================================================
# Reference config
# ============================================================

USE_REFERENCE_RAMP = True
REFERENCE_RAMP_TIME = 10.0

USE_TRAJECTORY_RAMP = True
TRAJECTORY_RAMP_TIME = 10.0

test_fixed_pose = False

target_phi_deg = -45.0
target_theta_deg = 40.0

# ============================================================
# Fixed-point sequence test
# ============================================================

USE_FIXED_POINT_SEQUENCE = False

FIXED_POINT_PHI_DEG = np.array([
    -45.0,
    0.0,
    45.0,
    90.0,
    135.0,
    -180.0,
    -135.0,
    -90.0,
])

FIXED_POINT_THETA_DEG = 35.0
POINT_HOLD_TIME = 20.0
POINT_TRANSITION_TIME = 30.0

# ============================================================
# IMU state filtering config
# ============================================================

THETA_PHI_VALID_DEG = HARDWARE_PARAMETERS["theta_phi_valid_deg"]
USE_IMU_VELOCITY = False
MAX_PHI_DOT_DEG_S = HARDWARE_PARAMETERS["max_phi_dot_deg_s"]
MAX_THETA_DOT_DEG_S = HARDWARE_PARAMETERS["max_theta_dot_deg_s"]
VELOCITY_FILTER_TAU = HARDWARE_PARAMETERS["velocity_filter_tau"]

# ============================================================
# IMU sanity guard
# ============================================================

USE_IMU_SANITY_GUARD = False
MAX_VALID_THETA_DEG = 45.0
MAX_THETA_JUMP_DEG = 12.0
MAX_PHI_JUMP_DEG = 60.0
MAX_CONSECUTIVE_BAD_IMU = 8

# ============================================================
# Pretension config
# ============================================================

pretension_before_capture = True
pretension_n = MPC_PARAMETERS.get("pretension", MPC_PARAMETERS["u_bound"][0])
pretension_ramp_steps = 50
pretension_ramp_dt = 0.0
pretension_settle_sec = 1.5

# ============================================================
# Hardware global objects
# ============================================================

imu = None
dxl = None
mapper = None
robot_hw = None

prev_phi = None
prev_theta = None
prev_time = None
phi_dot_f = 0.0
theta_dot_f = 0.0
prev_bend = None
bend_dot_f = np.zeros(2, dtype=float)
invalid_imu_count = 0


def smoothstep01(a):
    a = np.clip(float(a), 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def wrap_angle_rad(a):
    return np.arctan2(np.sin(a), np.cos(a))


def wrap_angle_2pi_rad(a):
    return np.mod(a, 2.0 * np.pi)


def rad_to_deg_signed(a):
    return float(np.rad2deg(wrap_angle_rad(a)))


def rad_to_deg_360(a):
    return float(np.rad2deg(wrap_angle_2pi_rad(a)))


def deg_to_rad_signed(a_deg):
    return wrap_angle_rad(np.deg2rad(a_deg))


def angular_error_rad(angle_rad, reference_rad):
    return wrap_angle_rad(angle_rad - reference_rad)


def angle_lerp_rad(start_rad, end_rad, alpha):
    alpha = np.asarray(alpha, dtype=float)
    return start_rad + alpha * wrap_angle_rad(end_rad - start_rad)


def phi_theta_to_bend(phi, theta):
    phi = np.asarray(phi, dtype=float)
    theta = np.maximum(np.asarray(theta, dtype=float), 0.0)
    return np.vstack((theta * np.cos(phi), theta * np.sin(phi)))


def bend_goal_from_phi_theta(phi_values, theta_values):
    bend_ref = phi_theta_to_bend(phi_values, theta_values)
    zeros = np.zeros_like(np.asarray(phi_values, dtype=float))
    return np.vstack((bend_ref, zeros, zeros))


def phi_theta_to_xyz(pcc_arm, phi, theta):
    q = np.array([float(phi), float(theta)], dtype=float)
    return np.array(pcc_arm.end_effector(q)).flatten()


def bend_sequence_to_xyz(pcc_arm, bend_sequence):
    return np.array([
        phi_theta_to_xyz(pcc_arm, phi, theta)
        for phi, theta in np.asarray(bend_sequence, dtype=float)
    ])


def build_fixed_point_bend_sequence(phi_deg_list, theta_deg):
    return np.column_stack((
        [deg_to_rad_signed(phi_deg) for phi_deg in phi_deg_list],
        np.full(len(phi_deg_list), np.deg2rad(theta_deg), dtype=float),
    ))


def fixed_point_sequence_bend_at_time(bend_points, t_sec, hold_time, transition_time):
    n_points = bend_points.shape[0]

    if n_points == 1:
        return bend_points[0].copy()

    t_sec = max(0.0, float(t_sec))

    if t_sec < hold_time:
        return bend_points[0].copy()

    tau = t_sec - hold_time
    block_time = transition_time + hold_time
    block_idx = int(tau // block_time)

    if block_idx >= n_points - 1:
        return bend_points[-1].copy()

    local_t = tau - block_idx * block_time
    start_phi, start_theta = bend_points[block_idx]
    end_phi, end_theta = bend_points[block_idx + 1]

    if local_t < transition_time:
        alpha = smoothstep01(local_t / transition_time)
        phi = angle_lerp_rad(start_phi, end_phi, alpha)
        theta = (1.0 - alpha) * start_theta + alpha * end_theta
        return np.array([phi, theta], dtype=float)

    return bend_points[block_idx + 1].copy()


def fixed_point_sequence_reference(bend_points, time_values, hold_time, transition_time):
    refs = np.array([
        fixed_point_sequence_bend_at_time(
            bend_points,
            tt,
            hold_time,
            transition_time,
        )
        for tt in time_values
    ])

    phi_values = refs[:, 0]
    theta_values = refs[:, 1]
    return bend_goal_from_phi_theta(phi_values, theta_values), phi_values, theta_values


def fixed_target_reference(start_phi, start_theta, target_phi, target_theta, time_values):
    if USE_REFERENCE_RAMP:
        alpha = np.clip(time_values / REFERENCE_RAMP_TIME, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        phi_values = angle_lerp_rad(start_phi, target_phi, alpha)
        theta_values = (1.0 - alpha) * start_theta + alpha * target_theta
    else:
        phi_values = np.full_like(time_values, target_phi, dtype=float)
        theta_values = np.full_like(time_values, target_theta, dtype=float)

    return bend_goal_from_phi_theta(phi_values, theta_values), phi_values, theta_values


def generate_bend_circle_trajectory(run_parameters, n_horizon):
    dt = run_parameters["dt"]
    total_time = run_parameters["T"]
    loop_time = run_parameters["T_loop"]

    points_per_loop = int(loop_time // dt)
    number_of_loops = int(np.ceil((total_time + n_horizon * dt) / loop_time)) + 10

    phi_start = deg_to_rad_signed(run_parameters.get("phi_start_deg", -45.0))
    theta = np.deg2rad(run_parameters.get("theta_trajectory_deg", 30.0))

    direction = run_parameters.get("direction", "cw")
    if direction == "cw":
        angle_end = -2.0 * np.pi
    else:
        angle_end = 2.0 * np.pi

    angles = np.linspace(0.0, angle_end, points_per_loop, endpoint=False)

    one_loop = np.column_stack((
        phi_start + angles,
        np.full(points_per_loop, theta, dtype=float),
    ))

    return np.tile(one_loop, (number_of_loops, 1)), one_loop

def sample_bend_trajectory_by_progress(bend_traj, progress_values):
    bend_traj = np.asarray(bend_traj, dtype=float)
    progress_values = np.asarray(progress_values, dtype=float)

    max_index = bend_traj.shape[0] - 1
    progress_values = np.clip(progress_values, 0.0, float(max_index))

    idx0 = np.floor(progress_values).astype(int)
    idx1 = np.minimum(idx0 + 1, max_index)
    alpha = (progress_values - idx0).reshape(-1, 1)

    return (1.0 - alpha) * bend_traj[idx0] + alpha * bend_traj[idx1]


def trajectory_reference(bend_traj, progress, start_phi, start_theta, time_values):
    horizon_progress = progress + np.arange(len(time_values))
    refs = sample_bend_trajectory_by_progress(bend_traj, horizon_progress)

    if USE_TRAJECTORY_RAMP:
        alpha = np.clip(time_values / TRAJECTORY_RAMP_TIME, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        phi_values = angle_lerp_rad(start_phi, refs[:, 0], alpha)
        theta_values = (1.0 - alpha) * start_theta + alpha * refs[:, 1]
    else:
        phi_values = refs[:, 0]
        theta_values = refs[:, 1]

    return bend_goal_from_phi_theta(phi_values, theta_values), phi_values, theta_values



def init_hardware():
    global imu, dxl, mapper, robot_hw

    robot_hw = RobotHardware(HARDWARE_PARAMETERS, MPC_PARAMETERS)
    robot_hw.init()
    imu = robot_hw.imu
    dxl = robot_hw.dxl
    mapper = robot_hw.mapper

def read_hardware_state():
    global prev_phi, prev_theta, prev_time
    global phi_dot_f, theta_dot_f
    global prev_bend, bend_dot_f
    global invalid_imu_count

    now = time.perf_counter()

    state = imu.get_state(
        phi_offset_deg=phi_offset_deg,
        phi_valid_thresh_deg=THETA_PHI_VALID_DEG,
    )

    phi = wrap_angle_rad(float(state["phi_rad"]))
    theta = float(state["theta_rad"])

    if theta < 0.0:
        theta = -theta
        phi = wrap_angle_rad(phi + np.pi)

    theta_valid = abs(np.rad2deg(theta)) >= THETA_PHI_VALID_DEG

    if theta_valid:
        if prev_phi is None:
            phi_used = wrap_angle_rad(phi)
        else:
            phi_used = prev_phi + wrap_angle_rad(phi - prev_phi)
    else:
        theta = 0.0
        if prev_phi is None:
            if USE_FIXED_POINT_SEQUENCE:
                phi_used = deg_to_rad_signed(FIXED_POINT_PHI_DEG[0])
            elif test_fixed_pose:
                phi_used = deg_to_rad_signed(target_phi_deg)
            else:
                phi_used = deg_to_rad_signed(RUN_PARAMETERS.get("phi_start_deg", -45.0))
        else:
            phi_used = prev_phi

    if USE_IMU_SANITY_GUARD and (prev_phi is not None) and (prev_theta is not None):
        theta_abs_deg = abs(np.rad2deg(theta))
        theta_jump_deg = abs(np.rad2deg(theta - prev_theta))
        phi_jump_deg = abs(np.rad2deg(wrap_angle_rad(phi_used - prev_phi)))

        imu_bad = (
            theta_abs_deg > MAX_VALID_THETA_DEG
            or theta_jump_deg > MAX_THETA_JUMP_DEG
            or (
                abs(np.rad2deg(prev_theta)) > THETA_PHI_VALID_DEG
                and phi_jump_deg > MAX_PHI_JUMP_DEG
            )
        )

        if imu_bad:
            invalid_imu_count += 1

            if invalid_imu_count == 1 or invalid_imu_count % 5 == 0:
                print(
                    "[IMU GUARD] rejected sample: "
                    f"phi={np.rad2deg(phi_used):+.2f} deg, "
                    f"theta={np.rad2deg(theta):+.2f} deg, "
                    f"theta_jump={theta_jump_deg:.2f} deg, "
                    f"phi_jump={phi_jump_deg:.2f} deg"
                )

            prev_time = now

            if invalid_imu_count >= MAX_CONSECUTIVE_BAD_IMU:
                raise RuntimeError("Too many consecutive invalid IMU samples.")

            return np.array([prev_phi, prev_theta, 0.0, 0.0], dtype=float)

        invalid_imu_count = 0

    bend = np.array(phi_theta_to_bend(phi_used, theta)).flatten()

    if prev_phi is None or prev_bend is None:
        phi_dot = 0.0
        theta_dot = 0.0
        bend_dot_f = np.zeros(2, dtype=float)
    else:
        dt_meas = max(now - prev_time, 1e-6)
        bend_dot_raw = (bend - prev_bend) / dt_meas

        max_bend_dot = np.deg2rad(MAX_THETA_DOT_DEG_S)
        bend_dot_norm = float(np.linalg.norm(bend_dot_raw))

        if bend_dot_norm > max_bend_dot:
            bend_dot_raw *= max_bend_dot / bend_dot_norm

        alpha = dt_meas / (VELOCITY_FILTER_TAU + dt_meas)
        bend_dot_f = (1.0 - alpha) * bend_dot_f + alpha * bend_dot_raw

        bx, by = bend
        bxdot, bydot = bend_dot_f
        theta_from_bend = float(np.hypot(bx, by))

        if theta_from_bend > 1e-8:
            theta_dot = float((bx * bxdot + by * bydot) / theta_from_bend)
        else:
            theta_dot = 0.0

        if theta_from_bend >= np.deg2rad(THETA_PHI_VALID_DEG):
            phi_dot_raw = float((bx * bydot - by * bxdot) / max(theta_from_bend**2, 1e-9))
        else:
            phi_dot_raw = 0.0

        phi_dot_raw = np.clip(
            phi_dot_raw,
            -np.deg2rad(MAX_PHI_DOT_DEG_S),
            np.deg2rad(MAX_PHI_DOT_DEG_S),
        )

        theta_dot = np.clip(
            theta_dot,
            -np.deg2rad(MAX_THETA_DOT_DEG_S),
            np.deg2rad(MAX_THETA_DOT_DEG_S),
        )

        phi_dot = 0.0
        theta_dot = float(theta_dot)

    phi_dot_f = phi_dot
    theta_dot_f = theta_dot
    prev_phi = phi_used
    prev_theta = theta
    prev_time = now
    prev_bend = bend.copy()

    if USE_IMU_VELOCITY:
        return np.array([phi_used, theta, phi_dot, theta_dot], dtype=float)

    return np.array([phi_used, theta, 0.0, 0.0], dtype=float)



def send_tensions_to_hardware(u_tendon):
    if robot_hw is None:
        raise RuntimeError("Hardware is not initialized. Call init_hardware() first.")

    u_tendon = np.asarray(u_tendon, dtype=float)
    u_tendon_clipped, currents_ma = robot_hw.send_tensions(
        u_tendon,
        use_friction=bool(HARDWARE_PARAMETERS.get("friction_enable", False)),
    )

    return u_tendon_clipped, currents_ma

def ramp_equal_pretension(target_tension_n, steps=50, dt=0.03):
    steps = max(1, int(steps))

    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        tensions = np.full(len(motor_ids), alpha * float(target_tension_n))
        send_tensions_to_hardware(tensions)
        time.sleep(float(dt))



def stop_hardware():
    global robot_hw

    try:
        if robot_hw is not None:
            robot_hw.stop()
            return
    except Exception:
        pass

    try:
        if dxl is not None:
            dxl.set_goal_current_mA([0.0] * len(motor_ids))
            time.sleep(0.1)
            dxl.torque_off()
    except Exception:
        pass

    try:
        if imu is not None:
            imu.close()
    except Exception:
        pass

    print("Hardware stopped.")

def main():
    start_time = time.perf_counter()
    num_iter = int(RUN_PARAMETERS["T"] / RUN_PARAMETERS["dt"])
    n_horizon = MPC_PARAMETERS["N"]

    pcc_arm = PCCSoftArm(ARM_PARAMETERS, RUN_PARAMETERS["dt"], num_iter)
    pcc_arm.true_current_state = RUN_PARAMETERS["x0"]
    pcc_arm.current_state = pcc_arm.true_current_state + pcc_arm.meas_error()

    target_phi_rad = deg_to_rad_signed(target_phi_deg)
    target_theta_rad = np.deg2rad(target_theta_deg)
    target_xyz = phi_theta_to_xyz(pcc_arm, target_phi_rad, target_theta_rad)

    fixed_point_bend_sequence = None
    fixed_point_xyz_sequence = None

    if USE_FIXED_POINT_SEQUENCE:
        fixed_point_bend_sequence = build_fixed_point_bend_sequence(
            FIXED_POINT_PHI_DEG,
            FIXED_POINT_THETA_DEG,
        )
        fixed_point_xyz_sequence = bend_sequence_to_xyz(pcc_arm, fixed_point_bend_sequence)

        print("Fixed-point sequence test in bend space:")
        for phi_deg, xyz in zip(FIXED_POINT_PHI_DEG, fixed_point_xyz_sequence):
            print(
                f"  phi={phi_deg:+7.1f} deg, "
                f"theta={FIXED_POINT_THETA_DEG:.1f} deg, "
                f"xyz={xyz}"
            )
    
    print("Experiment A: bend-space NMPC")
    print("target phi signed deg =", rad_to_deg_signed(target_phi_rad))
    print("target phi 0-360 deg  =", rad_to_deg_360(target_phi_rad))
    print("target theta deg      =", target_theta_deg)
    print("target xyz for plot   =", target_xyz)

    sigma_signed_deg = [rad_to_deg_signed(s) for s in ARM_PARAMETERS["sigma_k"]]
    sigma_360_deg = [rad_to_deg_360(s) for s in ARM_PARAMETERS["sigma_k"]]
    print("tendon sigma signed deg =", sigma_signed_deg)
    print("tendon sigma 0-360 deg  =", sigma_360_deg)

    bend_circular_traj, bend_plot_loop = generate_bend_circle_trajectory(
        RUN_PARAMETERS,
        n_horizon,
    )
    dotted_plotting_traj = bend_sequence_to_xyz(pcc_arm, bend_plot_loop)

    tf = n_horizon * RUN_PARAMETERS["dt"]
    ocp_solver = setup_ocp_solver(
        pcc_arm,
        MPC_PARAMETERS,
        n_horizon,
        tf,
        ACADOS_PARAMETERS,
    )
    init_hardware()

    current_tension_state = np.full(
        3 * pcc_arm.num_segments,
        pretension_n,
        dtype=float,
    )

    tension_ref_for_mpc = np.full(
        3 * pcc_arm.num_segments,
        MPC_PARAMETERS["u_bound"][0],
        dtype=float,
    )

    opti_index = [0]
    loop_time = np.zeros(num_iter)
    bend_debug = []

    ramp_start_phi = None
    ramp_start_theta = None
    trajectory_progress = 0.0

    try:
        with tqdm(
            total=num_iter * RUN_PARAMETERS["dt"],
            desc="MPC loop",
            bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {postfix}]",
        ) as pbar:
            for t in range(num_iter):
                try:
                    loop_time_0 = time.perf_counter()

                    x_meas = read_hardware_state()
                    pcc_arm.current_state = x_meas
                    pcc_arm.true_current_state = x_meas

                    if ramp_start_phi is None:
                        ramp_start_phi = float(x_meas[0])
                        ramp_start_theta = float(x_meas[1])

                    horizon_times = (
                        t * RUN_PARAMETERS["dt"]
                        + np.arange(n_horizon + 1) * RUN_PARAMETERS["dt"]
                    )

                    if test_fixed_pose:
                        if USE_FIXED_POINT_SEQUENCE:
                            bend_goal, phi_ref_values, theta_ref_values = fixed_point_sequence_reference(
                                fixed_point_bend_sequence,
                                horizon_times,
                                POINT_HOLD_TIME,
                                POINT_TRANSITION_TIME,
                            )
                        else:
                            bend_goal, phi_ref_values, theta_ref_values = fixed_target_reference(
                                ramp_start_phi,
                                ramp_start_theta,
                                target_phi_rad,
                                target_theta_rad,
                                horizon_times,
                            )
                    else:
                        bend_goal, phi_ref_values, theta_ref_values = trajectory_reference(
                            bend_circular_traj,
                            trajectory_progress,
                            ramp_start_phi,
                            ramp_start_theta,
                            horizon_times,
                        )

                    current_ref_phi_rad = float(phi_ref_values[0])
                    current_ref_theta_rad = float(theta_ref_values[0])
                    current_ref_xyz = phi_theta_to_xyz(
                        pcc_arm,
                        current_ref_phi_rad,
                        current_ref_theta_rad,
                    )

                    adapt_param = np.zeros(pcc_arm.num_adaptive_params)
                    x_aug_meas = np.hstack((x_meas, current_tension_state))

                    v0, x1, acados_status = mpc_step_acados(
                        ocp_solver,
                        x_aug_meas,
                        bend_goal,
                        adapt_param,
                        n_horizon,
                        MPC_PARAMETERS["u_bound"],
                        tension_ref=tension_ref_for_mpc,
                        return_status=True,
                    )

                    loop_time_1 = time.perf_counter()

                    v0 = np.clip(
                        v0,
                        MPC_PARAMETERS.get("v_bound", [-0.5, 0.5])[0],
                        MPC_PARAMETERS.get("v_bound", [-0.5, 0.5])[1],
                    )

                    u0 = current_tension_state + RUN_PARAMETERS["dt"] * v0
                    u0 = np.clip(
                        u0,
                        MPC_PARAMETERS["u_bound"][0],
                        MPC_PARAMETERS["u_bound"][1],
                    )

                    u_cmd, currents_ma = send_tensions_to_hardware(u0)
                    current_tension_state = u_cmd.copy()

                    tip_xyz_meas = phi_theta_to_xyz(pcc_arm, x_meas[0], x_meas[1])
                    bend_meas = np.array(phi_theta_to_bend(x_meas[0], x_meas[1])).flatten()
                    bend_ref = np.array(phi_theta_to_bend(current_ref_phi_rad, current_ref_theta_rad)).flatten()
                    bend_error = bend_ref - bend_meas
                    bend_error_norm = float(np.linalg.norm(bend_error))
                    tip_error_norm = float(np.linalg.norm(current_ref_xyz - tip_xyz_meas))

                    if DEBUG_PRINT and (t % DEBUG_EVERY == 0):
                        phi_deg_signed = rad_to_deg_signed(x_meas[0])
                        phi_deg_360 = rad_to_deg_360(x_meas[0])
                        target_phi_deg_signed = rad_to_deg_signed(current_ref_phi_rad)
                        target_phi_deg_360 = rad_to_deg_360(current_ref_phi_rad)
                        phi_err_deg = rad_to_deg_signed(
                            angular_error_rad(x_meas[0], current_ref_phi_rad)
                        )

                        theta_deg = np.rad2deg(x_meas[1])
                        theta_ref_deg = np.rad2deg(current_ref_theta_rad)
                        phi_dot_deg = np.rad2deg(x_meas[2])
                        theta_dot_deg = np.rad2deg(x_meas[3])

                        tqdm.write(
                            f"[t={t * RUN_PARAMETERS['dt']:.2f}s] "
                            f"phi={phi_deg_signed:+7.2f} deg "
                            f"({phi_deg_360:6.2f} deg360, "
                            f"ref={target_phi_deg_signed:+7.2f}/{target_phi_deg_360:6.2f} deg360, "
                            f"err={phi_err_deg:+7.2f}), "
                            f"theta={theta_deg:+7.2f} deg "
                            f"(ref={theta_ref_deg:+7.2f}), "
                            f"phi_dot={phi_dot_deg:+7.2f} deg/s, "
                            f"theta_dot={theta_dot_deg:+7.2f} deg/s | "
                            f"bend_ref=[{bend_ref[0]:+.3f}, {bend_ref[1]:+.3f}] | "
                            f"bend_meas=[{bend_meas[0]:+.3f}, {bend_meas[1]:+.3f}] | "
                            f"T_N=[{u_cmd[0]:.3f}, {u_cmd[1]:.3f}, {u_cmd[2]:.3f}] | "
                            f"Tref=[{tension_ref_for_mpc[0]:.3f}, {tension_ref_for_mpc[1]:.3f}, {tension_ref_for_mpc[2]:.3f}] | "
                            f"dTdt=[{v0[0]:+.3f}, {v0[1]:+.3f}, {v0[2]:+.3f}] N/s | "
                            f"I_mA=[{currents_ma[0]:+.1f}, {currents_ma[1]:+.1f}, {currents_ma[2]:+.1f}] | "
                            f"bend_err={bend_error_norm:.4f} rad | "
                            f"tip_err_plot={tip_error_norm:.4f} m"
                        )

                    pcc_arm.log_history(
                        u_cmd,
                        x1,
                        xyz_ref_current=current_ref_xyz,
                        u_rate=v0,
                    )

                    bend_debug.append([
                        t * RUN_PARAMETERS["dt"],
                        float(x_meas[0]),
                        float(x_meas[1]),
                        float(current_ref_phi_rad),
                        float(current_ref_theta_rad),
                        float(bend_meas[0]),
                        float(bend_meas[1]),
                        float(bend_ref[0]),
                        float(bend_ref[1]),
                        float(bend_error_norm),
                        float(acados_status),
                    ])

                    if not test_fixed_pose:
                        trajectory_progress = min(
                            trajectory_progress + 1.0,
                            float(bend_circular_traj.shape[0] - n_horizon - 2),
                        )

                    loop_time_2 = time.perf_counter()
                    total_time = (loop_time_2 - loop_time_0) * 1000.0
                    mpc_time = (loop_time_1 - loop_time_0) * 1000.0
                    send_time = (loop_time_2 - loop_time_1) * 1000.0
                    loop_time[t] = total_time

                    pbar.set_postfix(
                        MPC=f"{mpc_time:.2f}ms",
                        Send=f"{send_time:.2f}ms",
                        Total=f"{total_time:.2f}ms",
                        refresh=True,
                    )
                    pbar.update(RUN_PARAMETERS["dt"])

                    remaining_time = RUN_PARAMETERS["dt"] - (time.perf_counter() - loop_time_0)
                    if remaining_time > 0:
                        time.sleep(remaining_time)

                except Exception:
                    print("status:", ocp_solver.get_status())
                    print("alpha:", ocp_solver.get_stats("alpha"))
                    print("qp_iter:", ocp_solver.get_stats("qp_iter"))
                    print("residuals:", ocp_solver.get_residuals())
                    traceback.print_exc()
                    break

    finally:
        stop_hardware()

    print("--- %s seconds ---" % (time.perf_counter() - start_time))

    save = True
    run_name = datetime.now().strftime("hardware_run_%Y%m%d_%H%M%S")
    out_dir = Path("csv_and_plots_adapt") / "hardware_tests" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if pcc_arm.history_index == 0:
        print("No history was logged, so plots are skipped.")
        return

    valid_steps = pcc_arm.history_index
    valid_time = np.arange(valid_steps) * pcc_arm.dt
    valid_loop_time = loop_time[:valid_steps]

    if save:
        config_text = (
            "experiment=A_bend_space\n"
            "cost_output=[theta*cos(phi),theta*sin(phi),phi_dot,theta_dot]\n"

            f"test_fixed_pose={test_fixed_pose}\n"
            f"USE_FIXED_POINT_SEQUENCE={USE_FIXED_POINT_SEQUENCE}\n"

            f"target_phi_deg={target_phi_deg}\n"
            f"target_theta_deg={target_theta_deg}\n"

            f"theta_trajectory_deg={RUN_PARAMETERS['theta_trajectory_deg']}\n"
            f"phi_start_deg={RUN_PARAMETERS['phi_start_deg']}\n"
            f"T_loop={RUN_PARAMETERS['T_loop']}\n"
            f"T={RUN_PARAMETERS['T']}\n"
            f"direction={RUN_PARAMETERS.get('direction', 'cw')}\n"

            f"FIXED_POINT_THETA_DEG={FIXED_POINT_THETA_DEG}\n"
            f"pretension_n={pretension_n}\n"
            f"u_bound={MPC_PARAMETERS['u_bound']}\n"
            f"v_bound={MPC_PARAMETERS['v_bound']}\n"
        )
        (out_dir / "experiment_config.txt").write_text(config_text)

        if len(bend_debug) > 0:
            np.savetxt(
                out_dir / "bend_reference_debug.csv",
                np.asarray(bend_debug, dtype=float),
                delimiter=",",
                header=(
                    "time_s,phi_meas_rad,theta_meas_rad,"
                    "phi_ref_rad,theta_ref_rad,"
                    "bend_x_meas,bend_y_meas,bend_x_ref,bend_y_ref,"
                    "bend_error_norm_rad,acados_status"
                ),
                comments="",
            )


    # ============================================================
    # Save one combined PCC log for comparison plotting
    # ============================================================

    def get_history_col_safe(arr, col, n):
        """
        Return one history column with exactly length n.
        If unavailable or shorter than n, pad with NaN.
        """
        out = np.full(n, np.nan, dtype=float)

        try:
            arr = np.asarray(arr, dtype=float)

            if arr.ndim == 1:
                values = arr[:n]
            else:
                values = arr[:n, col]

            m = min(len(values), n)
            out[:m] = values[:m]

        except Exception as e:
            print(f"Could not add history column {col}: {e}")

        return out


    bend_arr = np.asarray(bend_debug, dtype=float)

    if bend_arr.size == 0:
        print("No bend_debug data, so combined PCC log was not saved.")
    else:
        n = len(bend_arr)

        loop_time_safe = np.full(n, np.nan, dtype=float)
        m_loop = min(len(loop_time), n)
        loop_time_safe[:m_loop] = loop_time[:m_loop]

        combined_log = {
            "time_s": bend_arr[:, 0],

            "phi_rad": bend_arr[:, 1],
            "theta_rad": bend_arr[:, 2],
            "phi_ref_rad": bend_arr[:, 3],
            "theta_ref_rad": bend_arr[:, 4],

            "bx_rad": bend_arr[:, 5],
            "by_rad": bend_arr[:, 6],
            "target_bx_rad": bend_arr[:, 7],
            "target_by_rad": bend_arr[:, 8],

            "bx_deg": np.rad2deg(bend_arr[:, 5]),
            "by_deg": np.rad2deg(bend_arr[:, 6]),
            "target_bx_deg": np.rad2deg(bend_arr[:, 7]),
            "target_by_deg": np.rad2deg(bend_arr[:, 8]),

            "bend_error_deg": np.rad2deg(bend_arr[:, 9]),

            # Total loop time, converted to seconds for compatibility
            "solve_time_s": loop_time_safe / 1000.0,
            "solve_time_ms": loop_time_safe,

            "acados_status": bend_arr[:, 10],

            "T1_N": get_history_col_safe(pcc_arm.history_u_tendon, 0, n),
            "T2_N": get_history_col_safe(pcc_arm.history_u_tendon, 1, n),
            "T3_N": get_history_col_safe(pcc_arm.history_u_tendon, 2, n),

            "dT1_rate_cmd_N_s": get_history_col_safe(pcc_arm.history_u_rate, 0, n),
            "dT2_rate_cmd_N_s": get_history_col_safe(pcc_arm.history_u_rate, 1, n),
            "dT3_rate_cmd_N_s": get_history_col_safe(pcc_arm.history_u_rate, 2, n),
        }

        # Debug check: all columns must have same length
        lengths = {k: len(np.asarray(v)) for k, v in combined_log.items()}
        print("Combined log column lengths:", lengths)

        combined_df = pd.DataFrame(combined_log)
        combined_df.to_csv(out_dir / "pcc_combined_log.csv", index=False)

        print("Saved combined PCC log:", out_dir / "pcc_combined_log.csv")

        plt.figure()
        plt.plot(valid_time, valid_loop_time)
        plt.title("Computation time per MPC step")
        plt.xlabel("Time [s]")
        plt.ylabel("Time [ms]")

        if save:
            plt.savefig(out_dir / "computation_time_per_MPC_step.png", dpi=200)

    print("Mean computation time per MPC step: ", np.mean(valid_loop_time), "ms")
    print("Max computation time per MPC step: ", np.max(valid_loop_time), "ms")
    print("Min computation time per MPC step: ", np.min(valid_loop_time), "ms")

    is_single_fixed_target = test_fixed_pose and not USE_FIXED_POINT_SEQUENCE

    history_plot(
        pcc_arm,
        MPC_PARAMETERS["u_bound"],
        xyz_traj=(
            fixed_point_xyz_sequence
            if USE_FIXED_POINT_SEQUENCE
            else (None if test_fixed_pose else dotted_plotting_traj)
        ),
        save=save,
        opti_index=opti_index,
        sim_parameters=RUN_PARAMETERS,
        target_phi_rad=target_phi_rad if is_single_fixed_target else None,
        target_theta_rad=target_theta_rad if is_single_fixed_target else None,
        target_xyz=target_xyz if is_single_fixed_target else None,
        current_gain_mA_per_N=gain_ma_per_n,
        current_offset_mA=offset_ma,
        current_limit_mA=i_max_ma,
        current_signs=tighten_sign,
        plot_adaptive=False,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
