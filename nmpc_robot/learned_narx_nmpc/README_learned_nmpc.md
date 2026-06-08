# Learned NARX-based NMPC Controller

This folder contains the hardware implementation of the learned NARX-based nonlinear model predictive controller (NMPC) for the tendon-driven continuum robot. The controller uses a trained NARX bend-space prediction model, acados for real-time optimization, Dynamixel motors in current-control mode, and a BNO055 IMU for feedback.

## Main files

```text
run_learned_nmpc_acados_shapes.py        Main recommended runner for fixed, circle, triangle, and star references
learned_nmpc_acados.py                   acados implementation of the learned NARX NMPC
learned_nmpc.py                          Learned NARX model utilities and older SciPy-based MPC
hardware_robot_friction.py               Hardware interface with current-level friction compensation
hardware_robot.py                        Basic hardware interface without friction compensation
save_results.py                          Saves logs and plots
params_learned_acados_friction_star.py   Parameters for star tracking
params_learned_acados_friction_triangle.py Parameters for triangle tracking
params_acados_friction.py                Parameters for circle/fixed acados run
params.py                                Parameters for older SciPy-based run
hardware/                               IMU, current mapping, and Dynamixel drivers
identified_models/                      Folder containing the trained NARX model
```

## Requirements

Install the required Python packages:

```bash
pip install numpy scipy matplotlib pandas casadi tqdm
```

A working acados Python installation is also required. The local `hardware/` folder and the trained model file in `identified_models/` must be present.

## Hardware and model settings

Before running, check the selected parameter file. The main hardware settings are:

```python
hardware = {
    "imu_port": "COM5",
    "dxl_port": "COM4",
    "motor_ids": [1, 2, 3],
    "gain_ma_per_n": 64.7,
    "i_max_ma": 270.0,
}
```

The learned model path is set using:

```python
model = {
    "path": "identified_models/stable_narx_ridge_bend_prbs_only_ny10.npz",
}
```

## Recommended way to run

Use the acados shapes runner:

```bash
python run_learned_nmpc_acados_shapes.py --params params_learned_acados_friction_star
```

or:

```bash
python run_learned_nmpc_acados_shapes.py --params params_learned_acados_friction_triangle
```

The `--params` argument can be either a module name or a `.py` file path.

At startup, the script initializes the hardware, applies pretension, asks the user to hold the arm straight, captures the straight IMU reference, and then starts the NMPC experiment.

## Changing experiment modes

The mode is changed in the selected parameter file by editing:

```python
run = {
    "mode": "...",
}
```

The recommended runner supports:

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
    "target_phi_deg": -45.0,
    "target_theta_deg": 30.0,
    "target_ramp_time": 10.0,
    "duration": 40.0,
}
```

### 2. Circular trajectory

```python
run = {
    "mode": "circle",
    "circle_theta_deg": 45.0,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 45.0,
    "target_ramp_time": 10.0,
    "duration": 55.0,
}
```

Use `"direction": "cw"` or `"ccw"`.

### 3. Triangle trajectory

```python
run = {
    "mode": "triangle",
    "triangle_theta_deg": 50.0,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 60.0,
    "triangle_smooth_edges": True,
    "target_ramp_time": 10.0,
    "duration": 70.0,
}
```

Run with:

```bash
python run_learned_nmpc_acados_shapes.py --params params_learned_acados_friction_triangle
```

### 4. Star trajectory

```python
run = {
    "mode": "star",
    "star_outer_theta_deg": 38.0,
    "star_inner_ratio": 0.48,
    "star_points": 5,
    "phi_start_deg": -45.0,
    "direction": "cw",
    "loop_time": 110.0,
    "star_smooth_edges": True,
    "target_ramp_time": 12.0,
    "duration": 125.0,
}
```

Run with:

```bash
python run_learned_nmpc_acados_shapes.py --params params_learned_acados_friction_star
```

## NMPC settings

The main NMPC settings are in the selected parameter file:

```python
mpc = {
    "pretension": 0.45,
    "u_min": 0.3,
    "u_max": 4.0,
    "horizon": 10,
    "max_du_step": 0.35,
    "qy": 350.0,
    "qf": 800.0,
    "rdu": 20.0,
    "ru": 0.08,
}
```

`max_du_step` is the maximum tendon-tension change per control step. The equivalent rate limit is:

```text
max tendon rate = max_du_step / dt
```

## Friction compensation

Friction compensation is configured in the `hardware` dictionary of the selected parameter file.

To enable or disable it:

```python
"friction_enable": True
```

The main friction settings are:

```python
"friction_direction_mode": "command_delta"
"friction_activation_mode": "hold_direction"
"friction_i_static_max_mA": 35.0
"friction_release_pulse_enable": True
```

The NMPC output remains tendon tension in Newtons. Friction compensation is only added after converting the tendon tension command to motor current.

## Safety settings

The safety settings are:

```python
safety = {
    "max_theta_deg": 80.0,
    "ramp_down_time": 3.0,
    "confirm_before_start": True,
}
```

If the measured bending angle exceeds `max_theta_deg`, the script stops the run and ramps the motors down.

## Output files

Each run creates a timestamped folder inside:

```text
hardware_validation/
```

The output folder is named:

```text
acados_narx_nmpc_<mode>_<timestamp>/
```

Typical output files include:

```text
log.csv
bend_tracking.png
bend_error.png
tensions.png
top_down_bend_space.png
phi_theta_tracking.png
tendon_rate_commanded.png
tendon_rate_actual.png
```

## Alternative runners

The following older/specialized runners are also included:

```bash
python run_nmpc.py
python run_nmpc_acados.py
python run_nmpc_acados_triangle.py
```

For most experiments, use:

```bash
python run_learned_nmpc_acados_shapes.py --params <parameter_file>
```
