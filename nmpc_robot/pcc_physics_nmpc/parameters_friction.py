import numpy as np

# material
E = 23e6
rho = 1220

# arm
r_o = 0.04 / 2
t_wall = 0.002
r_i = r_o - t_wall
r_d = 0.04 / 2
L = 0.315

A = np.pi * (r_o**2 - r_i**2)
I = np.pi * (r_o**4 - r_i**4) / 4
m = rho * np.pi * (r_o**2) * L * 0.5

print("Mass of each segment:", m)

k_phi = 0.0
k_theta = 0.2
beta = 0.05

rho_water = 1000
rho_air = 1.225
rho_liquid = rho_water

horizon_time = 1.0
dt = 0.1
num_segments = 1

tension_min_n = 0.3
tension_max_n = 4.0

q_bend = 300.0
q_phi_dot = 0.2
q_theta_dot = 3.0

qf_bend = 600.0
qf_phi_dot = 0.2
qf_theta_dot = 3.0

MPC_PARAMETERS = {
    "N": int(np.ceil(horizon_time / dt)),

    "Q": np.diag([
        500.0,
        500.0,
        1.0,
        6.0,
    ]),

    "Qf": np.diag([
        800.0,
        800.0,
        1.0,
        6.0,
    ]),

    "R": np.diag([
        10.0,
        10.0,
        10.0,
    ]),

    "R_rate": np.diag([
        0.08,
        0.08,
        0.08,
    ]),

    "u_bound": [0.3, 4.0],
    "v_bound": [-3.5, 3.5],

    "pretension": 0.45,
    "u_min": 0.3,
    "u_max": 4.0,
}



circle_theta_deg = 50.0
circle_loop_time = 45.0
circle_duration = 55.0
circle_phi_start_deg = -45.0
circle_direction = "cw"

theta_circle = np.deg2rad(circle_theta_deg)

RUN_PARAMETERS = {
    "dt": dt,
    "T": circle_duration,

    "x0": np.array([
        0.0,
        0.0,
        0.0,
        0.0,
    ]),

    "T_loop": circle_loop_time,
    "shape": "circle",

    "radius_trajectory": L * (1.0 - np.cos(theta_circle)) / theta_circle,
    "center_trajectory": np.array([
        0.0,
        0.0,
        L * np.sin(theta_circle) / theta_circle,
    ]),
    "rotation_angles_trajectory": np.array([
        np.deg2rad(0.0),
        np.deg2rad(0.0),
        np.deg2rad(circle_phi_start_deg),
    ]),

    "theta_trajectory_deg": circle_theta_deg,
    "phi_start_deg": circle_phi_start_deg,
    "direction": circle_direction,
}


sigma_k = np.deg2rad([
    -170.0,
    -58.0,
    62.0,
]).tolist()

ARM_PARAMETERS = {
    "L_segs": [L] * num_segments,
    "r_o": r_o,
    "r_i": r_i,
    "sigma_k": sigma_k,
    "rho_arm": rho,
    "beta": [beta] * num_segments,
    "K": np.diag([k_phi, k_theta] * num_segments),
    "num_segments": num_segments,
    "rho_liquid": rho_liquid,
    "r_d": r_d,
    "m": m,
    "tension_pretension": tension_min_n,
}

# ============================================================
# Hardware parameters 
# ============================================================

HARDWARE_PARAMETERS = {
    "imu_port": "COM5",
    "imu_baudrate": 460800,
    "dxl_port": "COM4",
    "dxl_baudrate": 1000000,
    "motor_ids": [1, 2, 3],

    # tension [N] -> current [mA]
    "gain_ma_per_n": 64.7,
    "offset_ma": 0.0,
    "tighten_sign": [-1.0, -1.0, -1.0],
    "i_max_ma": 270.0,

    "phi_offset_deg": 0.0,
    "theta_phi_valid_deg": 3.0,
    "velocity_filter_tau": 0.12,
    "max_phi_dot_deg_s": 300.0,
    "max_theta_dot_deg_s": 300.0,

    "current_stream_enable": True,
    "current_stream_hz": 50.0,
    "current_slew_rate_mA_s": 2400.0,

    "friction_enable": True,
    "friction_i_static_max_mA": 35.0,
    "friction_i_dynamic_mA": 0.0,
    "friction_velocity_decay_shape": "squared",
    "friction_velocity_decay_rad_s": 0.70,
    "friction_velocity_deadband_rad_s": 0.05,
    "friction_velocity_smooth_rad_s": 0.06,
    "friction_velocity_current_sign": [1.0, 1.0, 1.0],

    "friction_direction_mode": "command_delta",
    "friction_activation_mode": "hold_direction",
    "friction_cmd_smooth_mA": 4.0,
    "friction_min_delta_mA": 0.5,
    "friction_hold_timeout_s": 0.50,

    "friction_tighten_scale": 1.0,
    "friction_release_scale": 0.15,
    "friction_rate_limit_mA_per_step": 3.0,

    "friction_dither_amp_mA": 0.0,
    "friction_dither_freq_hz": 3.0,

    "friction_release_pulse_enable": True,
    "friction_release_pulse_mode": "additive",
    "friction_release_pulse_mA": 6.0,
    "friction_release_pulse_time_s": 0.06,
    "friction_release_pulse_min_dI_mA": 1.5,
    "friction_release_pulse_cooldown_s": 0.20,

    "friction_active_unwind_max_mA": 6.0,
}

ACADOS_PARAMETERS = {
    "code_export_directory": "c_generated_code_pcc_ocp",
    "solver_name": "pcc_arm_ocp",
    "nlp_solver_type": "SQP",
    "nlp_solver_max_iter": 100,
    "qp_solver": "PARTIAL_CONDENSING_HPIPM",
    "hessian_approx": "GAUSS_NEWTON",
    "qp_solver_iter_max": 100,
    "qp_solver_warm_start": 1,
    "regularize_method": "CONVEXIFY",
    "levenberg_marquardt": 1e-2,
    "globalization": "MERIT_BACKTRACKING",
    "globalization_use_SOC": False,
    "print_level": 0,

    "integrator_type": "IRK",

    "nlp_solver_tol_stat": 1e-4,
    "nlp_solver_tol_eq": 1e-6,
    "nlp_solver_tol_ineq": 1e-6,
    "nlp_solver_tol_comp": 1e-6,
}