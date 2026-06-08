from __future__ import annotations

import threading
import time

import numpy as np

from hardware.phi_theta import IMUBendReader
from hardware.current_actuator import CurrentTensionMapper
from hardware.dynamixel_controller import DynamixelController, BaseModel

from graybox_model import phi_theta_to_bend, wrap_angle_rad


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

        n_motors = len(self.cfg["motor_ids"])

        self.prev_nominal_currents_mA = np.zeros(n_motors, dtype=float)
        self._friction_direction = np.zeros(n_motors, dtype=float)
        self._prev_friction_currents_mA = np.zeros(n_motors, dtype=float)
        self._friction_last_update_time = np.zeros(n_motors, dtype=float)
        self._release_pulse_until = np.zeros(n_motors, dtype=float)
        self._release_pulse_last_trigger_time = np.zeros(n_motors, dtype=float)

        self.last_friction_direction = np.zeros(n_motors, dtype=float)
        self.last_friction_activation = np.zeros(n_motors, dtype=float)
        self.last_friction_speed_decay = np.zeros(n_motors, dtype=float)
        self.last_release_pulse_active = np.zeros(n_motors, dtype=float)

        self._stream_start_time = time.perf_counter()

        self.last_nominal_currents_mA = np.zeros(n_motors, dtype=float)
        self.last_friction_currents_mA = np.zeros(n_motors, dtype=float)
        self.last_sent_currents_mA = np.zeros(n_motors, dtype=float)
        self.last_motor_position_rad = np.zeros(n_motors, dtype=float)
        self.last_motor_velocity_rad_s = np.zeros(n_motors, dtype=float)
        self.last_measured_current_mA = np.zeros(n_motors, dtype=float)
        self.last_pwm_percent = np.zeros(n_motors, dtype=float)

        self._stream_enabled = bool(self.cfg.get("current_stream_enable", True))
        self._stream_hz = float(self.cfg.get("current_stream_hz", 50.0))
        self._stream_dt = 1.0 / max(self._stream_hz, 1.0)
        self._stream_stop_event = threading.Event()
        self._stream_thread = None
        self._stream_lock = threading.Lock()

        self._target_tensions = np.zeros(n_motors, dtype=float)
        self._target_use_friction = False
        self._stream_current_cmd_mA = np.zeros(n_motors, dtype=float)

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

        if self._stream_enabled:
            self._start_current_streamer()

        pretension = float(self.mpc_cfg["pretension"])
        print(f"Ramping equal pretension to {pretension:.3f} N per tendon...")
        self.ramp_equal_pretension(pretension, steps=50, dt=0.03)

        print("Hold the arm physically straight under pretension.")
        input("Press Enter to capture the straight IMU reference... ")
        self.imu.capture_straight_reference(n=80, dt=0.01)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _cfg_array(self, key: str, default: float, n: int) -> np.ndarray:
        value = self.cfg.get(key, default)
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=float)
        if arr.size != n:
            raise ValueError(f"hardware['{key}'] must be scalar or length {n}, got length {arr.size}")
        return arr.reshape(n)

    def _read_motor_info_safe(self):
        n = len(self.cfg["motor_ids"])
        try:
            pos, vel, current, pwm = self.dxl.read_info_with_unit(
                pwm_unit="percent",
                angle_unit="rad",
                current_unit="mA",
                fast_read=True,
            )
            return (
                np.asarray(pos, dtype=float).reshape(n),
                np.asarray(vel, dtype=float).reshape(n),
                np.asarray(current, dtype=float).reshape(n),
                np.asarray(pwm, dtype=float).reshape(n),
            )
        except Exception:
            return (
                self.last_motor_position_rad.copy(),
                self.last_motor_velocity_rad_s.copy(),
                self.last_measured_current_mA.copy(),
                self.last_pwm_percent.copy(),
            )

    # ---------------------------------------------------------------------
    # Friction compensation
    # ---------------------------------------------------------------------

    def _friction_compensation_mA(self, nominal_currents_mA, motor_velocity_rad_s):
        """Tendon-safe low-speed current friction compensation.

        """
        nominal_currents_mA = np.asarray(nominal_currents_mA, dtype=float)
        motor_velocity_rad_s = np.asarray(motor_velocity_rad_s, dtype=float)
        n = nominal_currents_mA.size

        if not bool(self.cfg.get("friction_enable", False)):
            self._friction_direction = np.zeros(n, dtype=float)
            self._prev_friction_currents_mA = np.zeros(n, dtype=float)
            self.last_friction_direction = np.zeros(n, dtype=float)
            self.last_friction_activation = np.zeros(n, dtype=float)
            self.last_friction_speed_decay = np.zeros(n, dtype=float)
            return np.zeros(n, dtype=float)

        i_static_max = self._cfg_array("friction_i_static_max_mA", 18.0, n)
        i_dynamic = self._cfg_array("friction_i_dynamic_mA", 0.0, n)
        velocity_decay = self._cfg_array("friction_velocity_decay_rad_s", 0.45, n)
        velocity_deadband = self._cfg_array("friction_velocity_deadband_rad_s", 0.05, n)
        velocity_smooth = self._cfg_array("friction_velocity_smooth_rad_s", 0.06, n)
        cmd_smooth = self._cfg_array("friction_cmd_smooth_mA", 8.0, n)
        min_delta = self._cfg_array("friction_min_delta_mA", 0.8, n)

        rate_limit = self._cfg_array("friction_rate_limit_mA_per_step", 1.5, n)

        direction_mode = str(self.cfg.get("friction_direction_mode", "command_delta")).lower()
        activation_mode = str(self.cfg.get("friction_activation_mode", "hold_direction")).lower()
        decay_shape = str(self.cfg.get("friction_velocity_decay_shape", "squared")).lower()

        dI = nominal_currents_mA - self.prev_nominal_currents_mA

        cmd_direction = np.tanh(dI / np.maximum(cmd_smooth, 1e-6))

        velocity_current_sign = self._cfg_array("friction_velocity_current_sign", 1.0, n)
        vel_direction = velocity_current_sign * np.tanh(
            motor_velocity_rad_s / np.maximum(velocity_smooth, 1e-6)
        )

        if direction_mode == "command_delta":
            desired_direction = cmd_direction
        elif direction_mode == "velocity":
            desired_direction = vel_direction
        elif direction_mode == "hybrid":
            desired_direction = np.where(
                np.abs(motor_velocity_rad_s) > velocity_deadband,
                vel_direction,
                cmd_direction,
            )
        else:
            raise ValueError(
                "friction_direction_mode must be 'command_delta', 'velocity', or 'hybrid'"
            )
        now = time.perf_counter()
        if direction_mode == "velocity":
            update_mask = np.abs(motor_velocity_rad_s) >= velocity_deadband
        else:
            update_mask = np.abs(dI) >= min_delta

        self._friction_direction = np.where(
            update_mask,
            desired_direction,
            self._friction_direction,
        )
        self._friction_last_update_time = np.where(
            update_mask,
            now,
            self._friction_last_update_time,
        )

        hold_timeout_s = float(self.cfg.get("friction_hold_timeout_s", 0.30))
        hold_active = (now - self._friction_last_update_time) <= max(hold_timeout_s, 0.0)

        if activation_mode == "command_delta":
            activation = np.clip(np.abs(dI) / np.maximum(cmd_smooth, 1e-6), 0.0, 1.0)
        elif activation_mode == "hold_direction":
            activation = np.where(
                hold_active,
                np.clip(np.abs(self._friction_direction), 0.0, 1.0),
                0.0,
            )
        elif activation_mode == "always":
            activation = np.ones(n, dtype=float)
        else:
            raise ValueError(
                "friction_activation_mode must be 'command_delta', 'hold_direction', or 'always'"
            )

        speed = np.abs(motor_velocity_rad_s)
        if decay_shape in ("squared", "stribeck", "gaussian"):
            speed_decay = np.exp(-np.square(speed / np.maximum(velocity_decay, 1e-6)))
        elif decay_shape in ("linear_exp", "exponential"):
            speed_decay = np.exp(-speed / np.maximum(velocity_decay, 1e-6))
        else:
            raise ValueError(
                "friction_velocity_decay_shape must be 'squared' or 'exponential'"
            )

        friction_mag = i_dynamic + (i_static_max - i_dynamic) * speed_decay

        tighten_scale = self._cfg_array("friction_tighten_scale", 1.0, n)
        release_scale = self._cfg_array("friction_release_scale", 0.20, n)
        tighten_sign = self._cfg_array("tighten_sign", -1.0, n)
        friction_sign = np.sign(self._friction_direction)

        tightening_help = (tighten_sign * friction_sign) > 0.0
        direction_scale = np.where(tightening_help, tighten_scale, release_scale)

        friction = (
            self._friction_direction
            * activation
            * friction_mag
            * direction_scale
        )

        dither_amp = self._cfg_array("friction_dither_amp_mA", 0.0, n)
        if np.any(dither_amp > 0.0):
            dither_freq = float(self.cfg.get("friction_dither_freq_hz", 3.0))
            near_stuck = (speed < velocity_deadband).astype(float)
            phase = 2.0 * np.pi * dither_freq * (
                time.perf_counter() - self._stream_start_time
            )
            friction += (
                near_stuck
                * np.sign(self._friction_direction)
                * dither_amp
                * np.sin(phase)
            )

        finite_mask = np.isfinite(rate_limit)
        if np.any(finite_mask):
            delta_fric = friction - self._prev_friction_currents_mA
            delta_fric = np.where(
                finite_mask,
                np.clip(delta_fric, -rate_limit, rate_limit),
                delta_fric,
            )
            friction = self._prev_friction_currents_mA + delta_fric

        self._prev_friction_currents_mA = friction.copy()
        self.last_friction_direction = self._friction_direction.copy()
        self.last_friction_activation = activation.copy()
        self.last_friction_speed_decay = speed_decay.copy()
        return friction

    def _compute_and_send_current_once(self, tensions, use_friction: bool):
        """Compute current command and write once to the Dynamixels."""
        u = np.asarray(tensions, dtype=float)
        u = np.clip(u, float(self.mpc_cfg["u_min"]), float(self.mpc_cfg["u_max"]))

        nominal_currents = self.mapper.tensions_to_signed_currents_mA(u)
        pos, motor_velocity, measured_current, pwm = self._read_motor_info_safe()

        if use_friction:
            friction_currents = self._friction_compensation_mA(nominal_currents, motor_velocity)
        else:
            friction_currents = np.zeros_like(nominal_currents)
            self._prev_friction_currents_mA = np.zeros_like(nominal_currents)

        target_currents = nominal_currents + friction_currents

        pulse_active = np.zeros_like(nominal_currents, dtype=bool)
        if use_friction and bool(self.cfg.get("friction_release_pulse_enable", False)):
            now = time.perf_counter()
            n = nominal_currents.size

            dI = nominal_currents - self.prev_nominal_currents_mA

            min_release_dI = self._cfg_array("friction_release_pulse_min_dI_mA", 2.0, n)
            vel_db = self._cfg_array("friction_velocity_deadband_rad_s", 0.05, n)
            tighten_sign = self._cfg_array("tighten_sign", -1.0, n)

            release_requested = (-tighten_sign * dI) > min_release_dI
            stuck = np.abs(motor_velocity) < vel_db

            pulse_time = float(self.cfg.get("friction_release_pulse_time_s", 0.05))
            pulse_current = self._cfg_array("friction_release_pulse_mA", 5.0, n)
            pulse_cooldown = float(self.cfg.get("friction_release_pulse_cooldown_s", 0.20))

            can_trigger = (now - self._release_pulse_last_trigger_time) >= max(
                pulse_cooldown,
                0.0,
            )
            trigger = release_requested & stuck & can_trigger

            self._release_pulse_until = np.where(
                trigger,
                now + pulse_time,
                self._release_pulse_until,
            )
            self._release_pulse_last_trigger_time = np.where(
                trigger,
                now,
                self._release_pulse_last_trigger_time,
            )

            pulse_active = now < self._release_pulse_until

            release_sign = -tighten_sign
            pulse_mode = str(self.cfg.get("friction_release_pulse_mode", "additive")).lower()

            if pulse_mode == "additive":
                pulse_target = target_currents + release_sign * pulse_current
            elif pulse_mode == "override":
                pulse_target = release_sign * pulse_current
            else:
                raise ValueError(
                    "friction_release_pulse_mode must be 'additive' or 'override'"
                )

            target_currents = np.where(
                pulse_active,
                pulse_target,
                target_currents,
            )

        self.last_release_pulse_active = pulse_active.astype(float)

        i_max = float(self.cfg["i_max_ma"])
        if use_friction:
            active_unwind_max = self._cfg_array("friction_active_unwind_max_mA", 5.0, nominal_currents.size)
            tighten_sign = self._cfg_array("tighten_sign", -1.0, nominal_currents.size)
            unwind_current = -tighten_sign * active_unwind_max
            too_much_unwind = (-tighten_sign * target_currents) > active_unwind_max
            target_currents = np.where(too_much_unwind, unwind_current, target_currents)

        target_currents = np.clip(target_currents, -i_max, i_max)

        rate_mA_s = float(self.cfg.get("current_slew_rate_mA_s", np.inf))
        if np.isfinite(rate_mA_s) and rate_mA_s > 0.0:
            max_step = rate_mA_s * self._stream_dt
            sent_currents = self._stream_current_cmd_mA + np.clip(
                target_currents - self._stream_current_cmd_mA,
                -max_step,
                max_step,
            )
        else:
            sent_currents = target_currents

        sent_currents = np.clip(sent_currents, -i_max, i_max)

        self.dxl.set_goal_current_mA(sent_currents)

        self.prev_nominal_currents_mA = nominal_currents.copy()
        self._stream_current_cmd_mA = sent_currents.copy()

        self.last_nominal_currents_mA = nominal_currents.copy()
        self.last_friction_currents_mA = friction_currents.copy()
        self.last_sent_currents_mA = sent_currents.copy()
        self.last_motor_position_rad = pos.copy()
        self.last_motor_velocity_rad_s = motor_velocity.copy()
        self.last_measured_current_mA = measured_current.copy()
        self.last_pwm_percent = pwm.copy()

        return u, sent_currents.copy()

    # ---------------------------------------------------------------------
    # Background current streamer
    # ---------------------------------------------------------------------

    def _start_current_streamer(self):
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stream_stop_event.clear()
        self._stream_start_time = time.perf_counter()
        self._stream_thread = threading.Thread(target=self._current_stream_loop, daemon=True)
        self._stream_thread.start()
        print(f"[RobotHardware] Current streamer enabled at {self._stream_hz:.1f} Hz")

    def _stop_current_streamer(self):
        self._stream_stop_event.set()
        if self._stream_thread is not None:
            self._stream_thread.join(timeout=1.0)
        self._stream_thread = None

    def _current_stream_loop(self):
        next_t = time.perf_counter()
        while not self._stream_stop_event.is_set():
            loop0 = time.perf_counter()

            with self._stream_lock:
                tensions = self._target_tensions.copy()
                use_friction = bool(self._target_use_friction)

            try:
                self._compute_and_send_current_once(tensions, use_friction=use_friction)
            except Exception:
                pass

            next_t += self._stream_dt
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)
            else:
                next_t = time.perf_counter()

    def send_tensions(self, tensions, use_friction: bool = True):
        """Set desired tendon tensions.
        """
        u = np.asarray(tensions, dtype=float)
        u = np.clip(u, float(self.mpc_cfg["u_min"]), float(self.mpc_cfg["u_max"]))

        if self._stream_enabled:
            with self._stream_lock:
                self._target_tensions = u.copy()
                self._target_use_friction = bool(use_friction)
            return u, self.last_sent_currents_mA.copy()

        return self._compute_and_send_current_once(u, use_friction=use_friction)

    def ramp_equal_pretension(self, target_tension: float, steps: int = 50, dt: float = 0.03):
        steps = max(1, int(steps))
        for k in range(steps):
            alpha = float(k + 1) / float(steps)
            u = np.full(len(self.cfg["motor_ids"]), alpha * float(target_tension), dtype=float)
            self.send_tensions(u, use_friction=False)
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
            if self._stream_enabled:
                self._stop_current_streamer()

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
