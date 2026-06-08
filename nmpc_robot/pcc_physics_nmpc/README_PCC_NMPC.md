# PCC-based NMPC Controller

This folder contains the code for running the PCC-based nonlinear model predictive controller (NMPC) on the tendon-driven continuum robot. The controller uses a constant-curvature model, acados for optimization, Dynamixel motors in current-control mode, and a BNO055 IMU for bend-angle feedback.

## Main files

```text
main_friction.py              Main hardware script with friction compensation
parameters_friction.py        MPC, trajectory, hardware, friction, and acados settings
pcc_arm.py                    PCC arm model and logging
utils.py                      PCC kinematics and dynamics
acados_utils.py               acados model and solver setup
hardware_robot_friction.py    Hardware interface and current-level friction compensation
visualisation_v2.py           Plotting and CSV export
hardware/                     Required IMU, current mapping, and Dynamixel drivers
```

## Requirements

Install the main Python packages:

```bash
pip install numpy matplotlib casadi tqdm
```

A working acados Python installation is also required. The local `hardware/` folder must be present.

## Hardware settings

Before running, check the hardware settings in `parameters_friction.py`:

```python
"imu_port": "COM5"
"dxl_port": "COM4"
"motor_ids": [1, 2, 3]
"gain_ma_per_n": 64.7
"i_max_ma": 270.0
```

These values should match the connected IMU and Dynamixel setup.

## How to run

From the folder containing the files, run:

```bash
python main_friction.py
```

At startup, the script initializes the motors and IMU, switches the motors to current-control mode, applies pretension, asks the user to hold the arm straight, captures the straight IMU reference, and then runs the NMPC experiment.

## Changing experiment modes

The experiment mode is selected in `main_friction.py`.

### 1. Circular trajectory tracking

This is the default mode.

```python
test_fixed_pose = False
USE_FIXED_POINT_SEQUENCE = False
```

The circular trajectory settings are changed in `parameters_friction.py`:

```python
circle_theta_deg = 60.0        # bending magnitude
circle_loop_time = 45.0        # time for one full circle
circle_duration = 55.0         # total run time
circle_phi_start_deg = -45.0   # starting bending direction
circle_direction = "cw"        # "cw" or "ccw"
```

### 2. Fixed target pose

To move the robot to one fixed bend pose:

```python
test_fixed_pose = True
USE_FIXED_POINT_SEQUENCE = False

target_phi_deg = -45.0
target_theta_deg = 40.0
```

The reference ramp can be enabled or disabled using:

```python
USE_REFERENCE_RAMP = True
REFERENCE_RAMP_TIME = 10.0
```

### 3. Fixed-point sequence

To move through a sequence of fixed bend directions:

```python
test_fixed_pose = True
USE_FIXED_POINT_SEQUENCE = True
```

Then edit the sequence in `main_friction.py`:

```python
FIXED_POINT_PHI_DEG = np.array([-45.0, 0.0, 45.0, 90.0])
FIXED_POINT_THETA_DEG = 35.0
POINT_HOLD_TIME = 20.0
POINT_TRANSITION_TIME = 30.0
```

## Friction compensation settings

Friction compensation is configured in `parameters_friction.py`.

To enable or disable it:

```python
"friction_enable": True
```

The main friction mode settings are:

```python
"friction_direction_mode": "command_delta"   # "command_delta", "velocity", or "hybrid"
"friction_activation_mode": "hold_direction" # "command_delta", "hold_direction", or "always"
"friction_release_pulse_enable": True
```

The NMPC still commands tendon tensions in Newtons. The friction compensation is added only at the motor-current level after converting tension to current.

## Output files

After each run, plots and CSV logs are saved in:

```text
csv_and_plots_adapt/hardware_tests/
```
