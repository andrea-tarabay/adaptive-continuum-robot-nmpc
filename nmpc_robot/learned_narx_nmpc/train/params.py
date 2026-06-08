from __future__ import annotations

hardware = {
    "imu_port": "COM5",
    "imu_baudrate": 460800,
    "dxl_port": "COM4",
    "dxl_baudrate": 1000000,
    "motor_ids": [1, 2, 3],

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

mpc = {
    "pretension": 0.45,
    "u_min": 0.45,
    "u_max": 4.0,
}

data_collection = {
    "dt": 0.10,
    "duration": 240.0,
    "pretension": 0.45,
    "amp": 2.55,
    "hold_s": 2.0,
    "max_tension": 3.0,
    "out_dir": "data/raw",
}