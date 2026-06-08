"""
Parameters for the acados learned NARX-NMPC hardware run.

Use with:
    python run_learned_nmpc_acados_shapes.py --params params_learned_acados_friction_star
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

model = {
    "path": "identified_models/stable_narx_ridge_bend_prbs_only_ny10.npz",
}

run = {
    "mode": "star",
    "dt": 0.10,
    "duration": 125.0,
    "target_ramp_time": 12.0,
    "out_dir": "hardware_validation",

    "target_phi_deg": -45.0,
    "target_theta_deg": 35.0,

    "star_outer_theta_deg": 38.0,
    "star_inner_ratio": 0.48,

    "star_points": 5,
    "phi_start_deg": -45.0,
    "direction": "cw",

    "loop_time": 110.0,
    "star_smooth_edges": True,
}

mpc = {
    "pretension": 0.45,
    "initial_bias": [0.50, 0.50, 0.50],
    "bias_ramp_time": 0.0,
    "settle_time": 0.0,

    "u_min": 0.35,
    "u_max": 4.0,

    "horizon": 10,
    "max_du_step": 0.35,

    "qy": 500.0,
    "qf": 850.0,
    "rdu": 35.0,
    "ru": 0.12,
}

safety = {
    "max_theta_deg": 80.0,
    "ramp_down_time": 3.0,
    "confirm_before_start": True,
}

acados = {
    "code_export_directory": "c_generated_code_narx_star_ocp",
    "solver_name": "narx_bend_ocp_star",

    "use_theta_constraint": False,
    "use_model_clipping": False,

    "nlp_solver_type": "SQP",
    "nlp_solver_max_iter": 20,
    "qp_solver": "PARTIAL_CONDENSING_HPIPM",
    "hessian_approx": "GAUSS_NEWTON",
    "qp_solver_iter_max": 50,
    "qp_solver_warm_start": 1,
    "regularize_method": "CONVEXIFY",
    "levenberg_marquardt": 1e-2,
    "globalization": "MERIT_BACKTRACKING",
    "print_level": 0,
}
