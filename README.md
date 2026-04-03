# Adaptive NMPC for a Tendon-Driven Continuum Robot


# Project Overview

This project builds a tendon-driven robotic arm control pipeline using an IMU for bend estimation, tendon/motor mapping, and Dynamixel current control.

## Core files

- `b1_phi_theta.py`  
  Reads the IMU quaternion data, captures a straight reference, and computes the bend direction `phi` and bend angle `theta`.

- `b2_current_actuator.py`  
  Converts tendon tension to motor current and also converts measured motor current back to estimated tendon tension/torque.

- `b3_tendon_mapping.py`  
  Converts bend error into a 2D bend command, then maps that command into 3 tendon tensions using the identified tendon directions.

- `dynamixel_controller.py`  
  Low-level Dynamixel communication file. Handles motor connection, operating modes, current commands, and feedback reading.

## Main control scripts

- `closed_loop_bend_current_control.py`  
  Main closed-loop controller for moving the arm to one desired bend target.  
  Flow: IMU -> bend error -> tendon tensions -> motor currents.

- `pid_Cl_live.py`  
  Similar closed-loop bend controller, but with PID terms and a live plot for tracking `phi` and `theta`.

- `ref_sequence.py`  
  Runs a full sequence of bend references instead of only one target. Also supports live plotting and saving performance plots.

## Identification / data collection

- `identify_current_vs_bend.py`  
  Runs a single-motor current sweep and logs how much bend is produced. Used to identify the relation between current and bending.

- `identify_current_vs_bend_v2_multiple.py`  
  Extended version of the previous script. Tests multiple motors and repeated runs, and stores all identification results in one session.

- `hangingmass_vs_bend.py`  
  Logs the arm bend produced by hanging masses on a tendon. Used to estimate the relation between applied load and bending.

- `test_blocks.py`  
  Simple dry-run script that tests the math pipeline without hardware control.  
  It shows how a desired bend becomes tensions, torques, and current commands.

## Plotting / post-processing

- `plotting.py`  
  Plots theta versus current from a single current-identification experiment.

- `plotting_v2_multiple.py`  
  Processes multi-motor identification results and produces aggregated plots such as theta vs current and bend direction per motor.

- `plotting_theta_vs_tension.py`  
  Processes hanging-mass experiment summaries and plots relations such as theta vs mass, theta vs force, phi vs force, and polar bend maps.

## File relationships

Main sensing and control chain:

`b1_phi_theta.py`  
-> gives measured bend state (`phi`, `theta`)

`b3_tendon_mapping.py`  
-> converts bend error into tendon tension commands

`b2_current_actuator.py`  
-> converts tendon tensions into motor current commands

`dynamixel_controller.py`  
-> sends those current commands to the motors and reads feedback

This chain is used mainly by:

- `closed_loop_bend_current_control.py`
- `pid_Cl_live.py`
- `ref_sequence.py`

Identification and analysis chain:

- `identify_current_vs_bend.py` -> `plotting.py`
- `identify_current_vs_bend_v2_multiple.py` -> `plotting_v2_multiple.py`
- `hangingmass_vs_bend.py` -> `plotting_theta_vs_tension.py`


