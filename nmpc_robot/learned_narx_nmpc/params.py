"""
Edit this file before running the learned NMPC.

Run from your project folder:
    python run_nmpc.py
"""

hardware = {
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
    "theta_phi_valid_deg": 8.0,
    "velocity_filter_tau": 0.12,
    "max_phi_dot_deg_s": 300.0,
    "max_theta_dot_deg_s": 300.0,
}

model = {
    "path": "identified_models/stable_narx_ridge_bend_prbs_only_ny10.npz",
}



run = {
    "mode": "circle",
    "dt": 0.10,
    "duration": 20.0,
    "target_ramp_time": 8.0,
    "out_dir": "hardware_validation",

    "target_phi_deg": -45.0,
    "target_theta_deg": 15.0,

    "circle_theta_deg": 30.0,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 45.0,
}
mpc = {
    "pretension": 0.45,
    "initial_bias": [0.45, 0.45, 0.45],
    "bias_ramp_time": 0.0,
    "settle_time": 0.0,
    "u_min": 0.35,
    "u_max": 3.2,

    "horizon": 6,
    "max_du_step": 0.12,
    "qy": 300.0,
    "qf": 1000.0,
    "rdu": 2.5,
    "ru": 0.0,
}
safety = {
    "max_theta_deg": 45.0,
    "ramp_down_time": 3.0,
    "confirm_before_start": True,
}