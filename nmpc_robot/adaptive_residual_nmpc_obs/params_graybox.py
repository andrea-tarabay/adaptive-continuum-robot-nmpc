
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

    "friction_enable": True,

    "friction_i_static_max_mA": 50.0,
    "friction_velocity_decay_rad_s": 0.8,
    "friction_cmd_smooth_mA": 10.0,
    "friction_min_delta_mA": 1.0,

    "friction_direction_mode": "command_delta",
    "friction_activation_mode": "command_delta",

    "friction_rate_limit_mA_per_step": float("inf"),
}

model = {
    "path": "identified_models/graybox_residual_bend_safe.npz",

    "tendon_angles_deg": [0.0, 120.0, 240.0],
    "tendon_radius_m": 1.0,  
    "tendon_lag_tau": 0.25,  
}

run = {
    "mode": "circle",  # "fixed" or "circle"
    "dt": 0.05,
    "duration": 40.0,
    "target_ramp_time": 10.0,
    "out_dir": "hardware_validation",

    "target_phi_deg": -45.0,
    "target_theta_deg": 20.0,

    "circle_theta_deg": 35.0,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 30.0,
}

mpc = {
    "pretension": 0.45,
    "initial_bias": [0.45, 0.45, 0.45],
    "bias_ramp_time": 0.0,
    "settle_time": 0.0,

    "u_min": 0.35,
    "u_max": 4.0,

    "horizon": 6,
    "max_du_step": 0.20,

    "qy": 500.0,
    "qf": 1000.0,
    "qtheta": 120.0,
    "rdu":  25.0,
    "ru": 0.25,
    "qdist": 0.001,
    "rdd": 280.0,
}
observer = {
    "velocity_filter_tau": 0.12,

    "disturbance_gain": 0.012,
    "disturbance_velocity_gain": 0.002,
    "disturbance_decay": 0.995,
    "disturbance_filter_tau": 1.1,
    "disturbance_clip": 0.08,
    "disturbance_rate_clip": 0.010,

    "innovation_deadband_deg": 0.6,
    "innovation_clip_deg": 4.0,
    "reset_on_large_jump_deg": 12.0,

    "freeze_disturbance_when_saturated": True,
    "saturation_margin_N": 0.03,

    "max_bend_dot_deg_s": 300.0,
}

safety = {
    "max_theta_deg": 70.0,
    "ramp_down_time": 3.0,
    "confirm_before_start": True,
}

data_collection = {
    "dt": 0.10,
    "out_dir": "data_graybox",
    "seed": 4,

    "u_min": 0.35,
    "u_max": 4.0,
    "pretension": 0.45,
    "max_du_step": 0.08,

    "static_hold_time": 3.0,
    "prbs_duration": 360.0,
    "sine_duration": 240.0,
    "circle_duration": 240.0,
    "prbs_hold_steps": 8,
    "prbs_amplitude_N": 0.65,

    "circle_amplitude_N": 0.65,
    "circle_period_s": 45.0,

    "confirm_before_start": True,
}
