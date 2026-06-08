
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

model = {
    "path": "identified_models/stable_narx_ridge_bend_prbs_only_ny10.npz",
}


run = {
    "mode": "circle",
    "dt": 0.10,
    "duration": 45.0,
    "target_ramp_time": 8.0,
    "out_dir": "hardware_validation",

    "target_phi_deg": -45.0,
    "target_theta_deg":  50.0,

    "circle_theta_deg": 50.0,
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
    "u_max": 4.0,

    "horizon": 6,
    "max_du_step": 0.20,
    "qy": 350.0,
    "qf": 1200.0,
    "rdu": 0.8,
    "ru": 0.08,
}

safety = {
    "max_theta_deg": 80.0,
    "ramp_down_time": 3.0,
    "confirm_before_start": True,
}

acados = {
    "code_export_directory": "c_generated_code_narx_ocp",
    "solver_name": "narx_bend_ocp",

    "use_theta_constraint": False,

    "use_model_clipping": False,

    "nlp_solver_type": "SQP",
    "nlp_solver_max_iter": 20,
    "levenberg_marquardt": 1e-2,
    "globalization": "MERIT_BACKTRACKING",
}
