# Comparative NMPC for Tendon-Driven Continuum Robot Control

**EPFL — CREATE Lab Semester Project**

This project compares three prediction models for **Nonlinear Model Predictive Control (NMPC)** of a tendon-driven continuum robot:

- Physics-based **Piecewise Constant Curvature (PCC)**
- Data-driven **NARX**
- **Adaptive residual gray-box** modeling

The goal was to study the trade-off between tracking accuracy, computational efficiency, and robustness to real hardware effects such as friction, hysteresis, and tendon slack.

📄 **[Full Project Report](docs/Comparative_NMPC_Report.pdf)**

---

## System

The controller was implemented and experimentally validated on a single-segment continuum robot actuated by three **Dynamixel XC330 motors**, with bend feedback from a tip-mounted **BNO055 IMU**.

```text
Reference Trajectory
        ↓
      NMPC
        ↓
Prediction Model
(PCC / NARX / Gray-Box)
        ↓
Tendon Commands
        ↓
Dynamixel Motors
        ↓
Continuum Robot
        ↓
    IMU Feedback
```

The controller runs at **10 Hz** with a 1-second prediction horizon and enforces tendon-tension and tendon-rate constraints.

---

## Models

### PCC
Physics-based baseline using a constant-curvature model of the robot.

### NARX
Data-driven dynamic model trained from experimental tendon commands and bend measurements.

### Adaptive Gray-Box
Combines a physics-inspired nominal model, a learned residual correction, and online disturbance estimation.

---

## Experimental Results

The controllers were evaluated on **circle, triangle, and star trajectories**.

For the star trajectory:

| Model | Mean Error | Mean Solve Time |
|---|---:|---:|
| NARX | **2.91°** | 43.8 ms |
| Adaptive Gray-Box | 4.79° | **10.4 ms** |

The NARX model achieved the best tracking accuracy, while the adaptive gray-box controller provided a strong compromise between accuracy and real-time computational efficiency.

---

## Main Contributions

- Implemented the NMPC framework
- Developed and compared PCC, NARX, and adaptive gray-box prediction models
- Collected experimental data and trained the NARX model
- Implemented online disturbance estimation
- Integrated NMPC with Dynamixel current control and IMU feedback
- Added friction compensation for improved hardware tracking
- Evaluated all controllers experimentally on the physical robot

---

## Technologies

**Python · CasADi · NMPC · System Identification · NARX · Gray-Box Modeling · Dynamixel · BNO055 IMU**

---

## Repository Structure

```text
continuum-robot-nmpc/
├── nmpc_robot/
│   └── Main NMPC implementation, including the prediction models,
│       controller logic, robot integration, and experimental evaluation.
│
├── setting up + pid control/
│   └── Hardware setup, Dynamixel configuration, IMU interfacing,
│       low-level PID/current control, and initial robot testing.
│
├── docs/
│   └── Comparative_NMPC_Report.pdf
│       Full project report with the modeling approach, controller formulation,
│       experimental setup, results, and discussion.
│
└── README.md
```

For the complete formulation, experiments, and analysis:

📄 **[Read the full report](docs/Comparative_NMPC_Report.pdf)**

---

**Andrea Tarabay**  
MSc Robotics, EPFL
