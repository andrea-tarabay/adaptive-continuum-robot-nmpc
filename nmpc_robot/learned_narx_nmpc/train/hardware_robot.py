from __future__ import annotations

import time

from pathlib import Path
import sys

# Allow running from the train/ folder while the hardware/ package stays one level above.
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

from hardware.phi_theta import IMUBendReader
from hardware.current_actuator import CurrentTensionMapper
from hardware.dynamixel_controller import DynamixelController, BaseModel

from learned_nmpc import phi_theta_to_bend, wrap_angle_rad


class RobotHardware:
    def __init__(self, hardware_cfg: dict, mpc_cfg: dict):
        self.cfg = hardware_cfg
        self.mpc_cfg = mpc_cfg

        self.imu = None
        self.dxl = None
        self.mapper = None

        self.prev_phi = None
        self.prev_theta = None
        self.prev_time = None
        self.prev_bend = None
        self.bend_dot_f = np.zeros(2, dtype=float)

    def init(self):
        self.imu = IMUBendReader(
            port=self.cfg["imu_port"],
            baudrate=self.cfg["imu_baudrate"],
            timeout=0.02,
        )

        motors = [BaseModel(motor_id=i) for i in self.cfg["motor_ids"]]
        self.dxl = DynamixelController(
            port_name=self.cfg["dxl_port"],
            motor_list=motors,
            baudrate=self.cfg["dxl_baudrate"],
            reverse_direction=False,
        )

        self.dxl.activate_controller()
        self.dxl.torque_off()
        self.dxl.set_operating_mode_all("current_control")
        self.dxl.torque_on()

        self.mapper = CurrentTensionMapper(
            gain_mA_per_N=self.cfg["gain_ma_per_n"],
            offset_mA=self.cfg["offset_ma"],
            tighten_sign=self.cfg["tighten_sign"],
            i_max_mA=self.cfg["i_max_ma"],
        )

        self.dxl.set_goal_current_mA([0.0] * len(self.cfg["motor_ids"]))
        time.sleep(0.3)

        pretension = float(self.mpc_cfg["pretension"])
        print(f"Ramping equal pretension to {pretension:.3f} N per tendon...")
        self.ramp_equal_pretension(pretension, steps=50, dt=0.03)

        print("Hold the arm physically straight under pretension.")
        input("Press Enter to capture the straight IMU reference... ")
        self.imu.capture_straight_reference(n=80, dt=0.01)

    def send_tensions(self, tensions):
        u = np.asarray(tensions, dtype=float)
        u = np.clip(u, float(self.mpc_cfg["u_min"]), float(self.mpc_cfg["u_max"]))
        currents = self.mapper.tensions_to_signed_currents_mA(u)
        self.dxl.set_goal_current_mA(currents)
        return u, np.asarray(currents, dtype=float)

    def ramp_equal_pretension(self, target_tension: float, steps: int = 50, dt: float = 0.03):
        steps = max(1, int(steps))
        for k in range(steps):
            alpha = float(k + 1) / float(steps)
            u = np.full(len(self.cfg["motor_ids"]), alpha * float(target_tension), dtype=float)
            self.send_tensions(u)
            time.sleep(float(dt))

    def read_state(self):
        now = time.perf_counter()
        state = self.imu.get_state(
            phi_offset_deg=self.cfg["phi_offset_deg"],
            phi_valid_thresh_deg=self.cfg["theta_phi_valid_deg"],
        )

        phi = wrap_angle_rad(float(state["phi_rad"]))
        theta = float(state["theta_rad"])

        if theta < 0.0:
            theta = -theta
            phi = wrap_angle_rad(phi + np.pi)

        if abs(np.rad2deg(theta)) < float(self.cfg["theta_phi_valid_deg"]):
            theta = 0.0
            if self.prev_phi is not None:
                phi = self.prev_phi

        bend = phi_theta_to_bend(phi, theta)

        if self.prev_bend is None or self.prev_time is None:
            phi_dot = 0.0
            theta_dot = 0.0
            self.bend_dot_f = np.zeros(2, dtype=float)
        else:
            dt_meas = max(now - self.prev_time, 1e-6)
            bend_dot_raw = (bend - self.prev_bend) / dt_meas

            max_bend_dot = np.deg2rad(float(self.cfg["max_theta_dot_deg_s"]))
            bend_dot_norm = float(np.linalg.norm(bend_dot_raw))
            if bend_dot_norm > max_bend_dot:
                bend_dot_raw *= max_bend_dot / bend_dot_norm

            tau = float(self.cfg["velocity_filter_tau"])
            alpha = dt_meas / (tau + dt_meas)
            self.bend_dot_f = (1.0 - alpha) * self.bend_dot_f + alpha * bend_dot_raw

            theta_from_bend = float(np.hypot(bend[0], bend[1]))
            if theta_from_bend > 1e-8:
                theta_dot = float((bend[0] * self.bend_dot_f[0] + bend[1] * self.bend_dot_f[1]) / theta_from_bend)
                phi_dot = float((bend[0] * self.bend_dot_f[1] - bend[1] * self.bend_dot_f[0]) / max(theta_from_bend**2, 1e-9))
            else:
                phi_dot = 0.0
                theta_dot = 0.0

            phi_dot = float(np.clip(
                phi_dot,
                -np.deg2rad(float(self.cfg["max_phi_dot_deg_s"])),
                np.deg2rad(float(self.cfg["max_phi_dot_deg_s"])),
            ))
            theta_dot = float(np.clip(
                theta_dot,
                -np.deg2rad(float(self.cfg["max_theta_dot_deg_s"])),
                np.deg2rad(float(self.cfg["max_theta_dot_deg_s"])),
            ))

        self.prev_phi = phi
        self.prev_theta = theta
        self.prev_time = now
        self.prev_bend = bend.copy()

        return phi, theta, phi_dot, theta_dot, bend

    def stop(self):
        try:
            if self.dxl is not None:
                self.dxl.set_goal_current_mA([0.0] * len(self.cfg["motor_ids"]))
                time.sleep(0.1)
                self.dxl.torque_off()
        except Exception:
            pass

        try:
            if self.imu is not None:
                self.imu.close()
        except Exception:
            pass

        print("Hardware stopped.")
