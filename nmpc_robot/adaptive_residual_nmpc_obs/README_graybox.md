# Adaptive Residual Gray-box NMPC

This folder contains the adaptive residual gray-box NMPC implementation for the tendon-driven continuum robot. The model combines a low-order physics-inspired bend-space model with a learned residual correction and an online disturbance observer. The controller commands tendon tensions in Newtons, which are converted to motor currents in the hardware layer.

## Main files

```text
run_graybox_nmpc_acados_shapes.py              Recommended hardware runner for fixed, circle, triangle, and star references
run_graybox_nmpc_acados.py                     acados hardware runner for fixed/circle references
run_graybox_nmpc.py                            older SciPy-based gray-box NMPC runner
graybox_model.py                               adaptive residual gray-box model and observer
graybox_model_acados.py                        acados solver wrapper for gray-box NMPC
hardware_robot_graybox_friction.py             hardware interface with current-level friction compensation
hardware_robot_graybox.py                      basic hardware interface without friction compensation
params_graybox.py                              parameters for training and older runner
params_graybox_acados_friction.py              parameters for acados fixed/circle runner
params_graybox_acados_friction_triangle_graybox.py  triangle experiment parameters
params_graybox_acados_friction_star_graybox.py      star experiment parameters
collect_data_graybox.py                        collects open-loop data for gray-box model identification
train_graybox_model.py                         trains the gray-box residual model
validate_graybox_model.py                      validates gray-box model rollouts
save_results.py                                saves hardware logs and plots
evaluate_one_nmpc_run.py                       evaluates one NMPC run
```

The `hardware/` folder must also be available because the scripts use the IMU and Dynamixel hardware drivers.

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy pandas matplotlib tqdm casadi
```

A working acados Python installation is required for the acados runners.

## Expected folders

```text
data_graybox/              raw gray-box data collection runs
identified_models/         saved gray-box model files
hardware_validation/       NMPC hardware run outputs
validation_graybox/        validation plots and metrics
```

## Hardware settings

Before running, check the hardware dictionary in the selected parameter file:

```python
hardware = {
    "imu_port": "COM5",
    "dxl_port": "COM4",
    "motor_ids": [1, 2, 3],
    "gain_ma_per_n": 64.7,
    "i_max_ma": 270.0,
}
```

The model path is usually:

```python
model = {
    "path": "identified_models/graybox_residual_bend_safe.npz",
}
```

## 1. Collect gray-box training data

Run:

```bash
python collect_data_graybox.py
```

The collection settings are in `params_graybox.py`:

```python
data_collection = {
    "dt": 0.10,
    "out_dir": "data_graybox",
    "u_min": 0.35,
    "u_max": 4.0,
    "pretension": 0.45,
    "max_du_step": 0.08,
    "static_hold_time": 3.0,
    "prbs_duration": 360.0,
    "sine_duration": 240.0,
    "circle_duration": 240.0,
}
```

The script collects several open-loop segments:

```text
static
prbs
sine
circle_tension
```

Each segment is saved as a CSV file inside a timestamped folder under `data_graybox/`.


## 2. Train the gray-box model

After collecting data, train the model with:

```bash
python train_graybox_model.py --data data_graybox --out identified_models/graybox_residual_bend_safe.npz
```

Useful options:

```text
--data                  CSV file or folder containing data-collection CSVs
--out                   output .npz model path
--dt                    training timestep
--tendon-lag-tau        first-order tendon lag time constant
--tendon-radius-m       effective tendon moment radius
--smooth-window         smoothing window for bend data
--phys-ridge            ridge weight for physics term
--res-ridge             ridge weight for residual term
--res-threshold         sparsification threshold
--residual-scale        scale applied to learned residual
--accel-clip            acceleration clipping limit
--max-bend-deg          maximum allowed bend angle
--max-bend-dot-deg-s    maximum allowed bend speed
```

The output model is saved as an `.npz` file and is later loaded by the NMPC scripts.

## 3. Validate the trained model

Run:

```bash
python validate_graybox_model.py --model identified_models/graybox_residual_bend_safe.npz --data data_graybox --out-dir validation_graybox
```

Optional reset window:

```bash
python validate_graybox_model.py --model identified_models/graybox_residual_bend_safe.npz --data data_graybox --reset-every 6
```

This produces validation plots and rollout metrics in `validation_graybox/`.

## 4. Recommended hardware run

For final hardware experiments, use the acados shapes runner:

```bash
python run_graybox_nmpc_acados_shapes.py --params params_graybox_acados_friction_triangle_graybox
```

or:

```bash
python run_graybox_nmpc_acados_shapes.py --params params_graybox_acados_friction_star_graybox
```

The `--params` argument can be a module name or a `.py` file path.

At startup, the script loads the gray-box model, builds or loads the acados solver, initializes the hardware, applies pretension, asks the user to hold the arm straight, captures the straight IMU reference, and then starts the NMPC run.

## Changing experiment modes

The experiment mode is selected in the `run` dictionary of the chosen parameter file:

```python
run = {
    "mode": "...",
}
```

The recommended shapes runner supports:

```text
fixed
circle
triangle
star
```

### 1. Fixed target pose

```python
run = {
    "mode": "fixed",
    "dt": 0.10,
    "duration": 40.0,
    "target_ramp_time": 10.0,
    "target_phi_deg": -45.0,
    "target_theta_deg": 35.0,
    "out_dir": "hardware_validation",
}
```

### 2. Circular trajectory

```python
run = {
    "mode": "circle",
    "dt": 0.10,
    "duration": 55.0,
    "target_ramp_time": 10.0,
    "circle_theta_deg": 45.0,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 45.0,
    "out_dir": "hardware_validation",
}
```

Use `"direction": "cw"` or `"ccw"`.

### 3. Triangle trajectory

```python
run = {
    "mode": "triangle",
    "dt": 0.10,
    "duration": 70.0,
    "target_ramp_time": 10.0,
    "triangle_theta_deg": 45.0,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 60.0,
    "triangle_smooth_edges": True,
    "out_dir": "hardware_validation",
}
```

Run with:

```bash
python run_graybox_nmpc_acados_shapes.py --params params_graybox_acados_friction_triangle_graybox
```

### 4. Star trajectory

```python
run = {
    "mode": "star",
    "dt": 0.10,
    "duration": 125.0,
    "target_ramp_time": 12.0,
    "star_outer_theta_deg": 38.0,
    "star_inner_ratio": 0.48,
    "star_points": 5,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 110.0,
    "star_smooth_edges": True,
    "out_dir": "hardware_validation",
}
```

Run with:

```bash
python run_graybox_nmpc_acados_shapes.py --params params_graybox_acados_friction_star_graybox
```

## NMPC settings

The main NMPC settings are in the selected parameter file:

```python
mpc = {
    "pretension": 0.45,
    "initial_bias": [0.50, 0.50, 0.50],
    "u_min": 0.35,
    "u_max": 4.0,
    "horizon": 10,
    "max_du_step": 0.35,
    "qy": 1000.0,
    "qf": 1500.0,
    "qtheta": 30.0,
    "rdu": 25.0,
    "ru": 8.0,
    "qdist": 0.5,
    "rdd": 1000.0,
}
```

`max_du_step` is the maximum tendon-tension increment per control step. The equivalent tendon-rate limit is:

```text
max tendon rate = max_du_step / dt
```

## Observer settings

The gray-box controller uses an online disturbance observer. The main settings are:

```python
observer = {
    "velocity_filter_tau": 0.18,
    "disturbance_gain": 0.004,
    "disturbance_velocity_gain": 0.0008,
    "disturbance_decay": 0.992,
    "disturbance_filter_tau": 2.0,
    "disturbance_clip": 0.035,
    "disturbance_rate_clip": 0.0025,
    "innovation_deadband_deg": 1.0,
    "innovation_clip_deg": 2.5,
    "reset_on_large_jump_deg": 8.0,
}
```

The observer estimates a small bend-acceleration disturbance used inside the NMPC prediction model.

## Friction compensation

Friction compensation is configured in the `hardware` dictionary:

```python
"friction_enable": True
"friction_direction_mode": "command_delta"
"friction_activation_mode": "hold_direction"
"friction_i_static_max_mA": 35.0
"friction_release_pulse_enable": True
```

The NMPC still outputs tendon tension in Newtons. Friction compensation is added only after the tension command is converted to motor current.

## Safety settings

The safety settings are:

```python
safety = {
    "max_theta_deg": 70.0,
    "ramp_down_time": 3.0,
    "confirm_before_start": True,
}
```

If the measured bend exceeds the maximum allowed angle, the run is stopped and the motors are ramped down.

## Output files

Each hardware run creates a timestamped folder inside:

```text
hardware_validation/
```

The output folder name has the form:

```text
adaptive_residual_graybox_acados_<mode>_<timestamp>/
```

Typical saved files include:

```text
log.csv
bend_tracking.png
bend_error.png
tensions.png
tension_rate_commands.png
effective_tensions.png
estimated_disturbance.png
top_down_bend_space.png
phi_theta_tracking.png
```

The log also includes solver status, solve time, observer diagnostics, effective tendon tension, and friction/current diagnostics when available.

## Evaluating hardware runs

Evaluate one run:

```bash
python evaluate_one_nmpc_run.py hardware_validation/<run_folder>
```

or:

```bash
python evaluate_one_nmpc_run.py hardware_validation/<run_folder>/log.csv --du-limit 0.35
```



## Alternative runners

The following runners are also included:

```bash
python run_graybox_nmpc.py
python run_graybox_nmpc_acados.py
```
