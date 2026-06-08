from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize


# -----------------------------
# Shared geometry/helper methods
# -----------------------------


def smoothstep01(a: float) -> float:
    a = np.clip(float(a), 0.0, 1.0)
    return float(a * a * (3.0 - 2.0 * a))


def wrap_angle_rad(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def phi_theta_to_bend(phi: float, theta: float) -> np.ndarray:
    """Map polar bend angle to Cartesian bend vector [bx, by]."""
    return np.array([theta * np.cos(phi), theta * np.sin(phi)], dtype=float)


def bend_to_phi_theta(b: np.ndarray) -> tuple[float, float]:
    bx, by = float(b[0]), float(b[1])
    theta = float(np.hypot(bx, by))
    phi = 0.0 if theta < 1e-9 else float(np.arctan2(by, bx))
    return phi, theta


def append_keep(arr: np.ndarray, row: np.ndarray, max_len: int) -> np.ndarray:
    arr = np.vstack([arr, np.asarray(row, dtype=float).reshape(1, -1)])
    if len(arr) > max_len:
        arr = arr[-max_len:]
    return arr


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip a vector by Euclidean norm without changing its direction."""
    v = np.asarray(v, dtype=float).copy()
    max_norm = float(max_norm)
    if max_norm <= 0.0 or not np.isfinite(max_norm):
        return v
    n = float(np.linalg.norm(v))
    if n > max_norm:
        v *= max_norm / max(n, 1e-12)
    return v


def fixed_reference(t: float, start_bend: np.ndarray, target_bend: np.ndarray, ramp_time: float) -> np.ndarray:
    a = smoothstep01(t / max(float(ramp_time), 1e-9))
    return (1.0 - a) * np.asarray(start_bend, dtype=float) + a * np.asarray(target_bend, dtype=float)


def circle_reference(
    t: float,
    start_bend: np.ndarray,
    theta_rad: float,
    phi_start_rad: float,
    loop_time: float,
    direction: str,
    ramp_time: float,
) -> np.ndarray:
    first = phi_theta_to_bend(phi_start_rad, theta_rad)
    if t < ramp_time:
        a = smoothstep01(t / max(float(ramp_time), 1e-9))
        return (1.0 - a) * np.asarray(start_bend, dtype=float) + a * first

    sign = -1.0 if str(direction).lower() == "cw" else 1.0
    tau = t - max(float(ramp_time), 0.0)
    phi = phi_start_rad + sign * 2.0 * np.pi * tau / max(float(loop_time), 1e-9)
    return phi_theta_to_bend(phi, theta_rad)


# -----------------------------
# Model settings and model class
# -----------------------------


@dataclass
class MpcSettings:
    horizon: int
    u_min: float
    u_max: float
    max_du_step: float
    qy: float
    qf: float
    rdu: float
    ru: float
    u_bias: np.ndarray
    max_bend_deg: float = 45.0
    qtheta: float = 0.0
    qdist: float = 0.0
    rdd: float = 0.0

@dataclass
class ObserverSettings:
    velocity_filter_tau: float = 0.12

    disturbance_gain: float = 0.012          # position innovation gain
    disturbance_velocity_gain: float = 0.004 # velocity innovation gain
    disturbance_decay: float = 0.995
    disturbance_filter_tau: float = 0.8
    disturbance_clip: float = 0.20           # rad/s^2, norm clip
    disturbance_rate_clip: float = 0.025     # rad/s^2 per update
    innovation_deadband_deg: float = 0.4
    innovation_clip_deg: float = 8.0
    freeze_disturbance_when_saturated: bool = False
    saturation_margin_N: float = 0.03
    reset_on_large_jump_deg: float = 35.0

    max_bend_dot_deg_s: float = 300.0


class AdaptiveResidualModel:
    """Low-order gray-box bend dynamics for a 3-tendon soft continuum robot.

    State layout:
        x = [bx, by, vx, vy, Tcmd1, Tcmd2, Tcmd3, Teff1, Teff2, Teff3, dx, dy]

    Control:
        dU = [dT1, dT2, dT3] tension increments over one control step.

    Dynamics:
        tension command is rate-controlled by NMPC;
        effective tendon tension follows first-order lag;
        bend acceleration = structured physics + sparse residual + online disturbance.
    """

    nx = 12
    nu = 3
    y_dim = 2

    def __init__(self, model_path: str | Path):
        self.path = str(model_path)
        data = np.load(self.path, allow_pickle=True)

        self.dt = float(data["dt"])
        self.pretension = float(data["pretension"])
        self.tendon_angles_rad = np.asarray(data["tendon_angles_rad"], dtype=float).reshape(3)
        self.tendon_radius_m = float(data.get("tendon_radius_m", 1.0))
        self.tendon_lag_tau = float(data.get("tendon_lag_tau", 0.18))
        self.alpha_u = float(data.get("alpha_u", self.dt / max(self.tendon_lag_tau, self.dt)))
        self.alpha_u = float(np.clip(self.alpha_u, 0.0, 1.0))

        # Nominal physics acceleration: a = W_phys @ [1,bx,by,vx,vy,mx,my]
        self.W_phys = np.asarray(data["W_phys"], dtype=float)  # shape (7,2)
        self.W_res = np.asarray(data["W_res"], dtype=float)    # shape (n_features,2)
        self.residual_scale = float(data.get("residual_scale", 1.0))
        self.accel_clip = float(data.get("accel_clip", 60.0))

        self.max_bend_rad = float(data.get("max_bend_rad", np.deg2rad(85.0)))
        self.max_bend_dot_rad_s = float(data.get("max_bend_dot_rad_s", np.deg2rad(500.0)))

        self.feature_names = [str(x) for x in data.get("feature_names", np.array([], dtype=object)).tolist()]

    def set_tendon_lag_tau(self, tau_s: float) -> None:
        """Override the tendon lag used by runtime predictions."""
        tau_s = float(tau_s)
        if not np.isfinite(tau_s) or tau_s <= 0.0:
            return
        self.tendon_lag_tau = tau_s
        self.alpha_u = float(1.0 - np.exp(-self.dt / max(tau_s, 1e-9)))
        self.alpha_u = float(np.clip(self.alpha_u, 0.0, 1.0))

    @property
    def u_bias(self) -> np.ndarray:
        return np.full(3, self.pretension, dtype=float)

    def tendon_moment(self, tensions: np.ndarray) -> np.ndarray:
        """Differential tendon tensions -> planar bend moment-like input [mx,my].

        Tendon angles are measured around the backbone. The common tension component
        mostly creates preload; the differential component creates bend.
        """
        T = np.asarray(tensions, dtype=float).reshape(3)
        dT = T - self.pretension
        dT = dT - np.mean(dT)
        c = np.cos(self.tendon_angles_rad)
        s = np.sin(self.tendon_angles_rad)
        mx = self.tendon_radius_m * float(np.dot(c, dT))
        my = self.tendon_radius_m * float(np.dot(s, dT))
        return np.array([mx, my], dtype=float)

    def physics_features(self, b: np.ndarray, v: np.ndarray, teff: np.ndarray) -> np.ndarray:
        m = self.tendon_moment(teff)
        return np.array([1.0, b[0], b[1], v[0], v[1], m[0], m[1]], dtype=float)

    def residual_features(self, b: np.ndarray, v: np.ndarray, tcmd: np.ndarray, teff: np.ndarray, dU: np.ndarray) -> np.ndarray:
        m = self.tendon_moment(teff)
        bx, by = float(b[0]), float(b[1])
        vx, vy = float(v[0]), float(v[1])
        mx, my = float(m[0]), float(m[1])
        du = np.asarray(dU, dtype=float).reshape(3)
        ue = np.asarray(teff, dtype=float).reshape(3) - self.pretension
        return np.array([
            1.0,
            bx, by, vx, vy, mx, my,
            du[0], du[1], du[2],
            ue[0], ue[1], ue[2],
            np.tanh(3.0 * vx), np.tanh(3.0 * vy),
            bx * vx, by * vy,
            bx * mx, bx * my, by * mx, by * my,
            vx * mx, vx * my, vy * mx, vy * my,
            bx * bx, by * by, vx * vx, vy * vy, mx * mx, my * my, mx * my,
        ], dtype=float)

    def acceleration(self, b: np.ndarray, v: np.ndarray, tcmd: np.ndarray, teff: np.ndarray, dU: np.ndarray, disturbance: np.ndarray | None = None) -> np.ndarray:
        f_phys = self.physics_features(b, v, teff)
        a_phys = f_phys @ self.W_phys
        f_res = self.residual_features(b, v, tcmd, teff, dU)
        a_res = self.residual_scale * (f_res @ self.W_res)
        a = a_phys + a_res
        if disturbance is not None:
            a = a + np.asarray(disturbance, dtype=float).reshape(2)
        return np.clip(a, -self.accel_clip, self.accel_clip)

    def make_state(self, bend: np.ndarray, tension_cmd: np.ndarray, bend_dot: np.ndarray | None = None, disturbance: np.ndarray | None = None) -> np.ndarray:
        b = np.asarray(bend, dtype=float).reshape(2)
        v = np.zeros(2, dtype=float) if bend_dot is None else np.asarray(bend_dot, dtype=float).reshape(2)
        u = np.asarray(tension_cmd, dtype=float).reshape(3)
        d = np.zeros(2, dtype=float) if disturbance is None else np.asarray(disturbance, dtype=float).reshape(2)
        return np.hstack([b, v, u, u, d]).astype(float)

    def step(self, x: np.ndarray, dU: np.ndarray, settings: MpcSettings | None = None, *, use_disturbance: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(self.nx)
        dU = np.asarray(dU, dtype=float).reshape(3)
        dt = self.dt

        b = x[0:2]
        v = x[2:4]
        u_cmd = x[4:7]
        u_eff = x[7:10]
        d = x[10:12]

        if settings is None:
            u_next = u_cmd + dU
        else:
            u_next = np.clip(u_cmd + dU, float(settings.u_min), float(settings.u_max))

        u_eff_next = u_eff + self.alpha_u * (u_next - u_eff)
        a = self.acceleration(b, v, u_next, u_eff_next, dU, d if use_disturbance else None)

        # Semi-implicit Euler 
        v_next = v + dt * a
        v_next = clip_norm(v_next, self.max_bend_dot_rad_s)

        b_next = b + dt * v_next
        b_next_clipped = clip_norm(b_next, self.max_bend_rad)
        if np.linalg.norm(b_next_clipped - b_next) > 1e-12:
            radial = b_next_clipped / max(float(np.linalg.norm(b_next_clipped)), 1e-12)
            outward_speed = float(v_next @ radial)
            if outward_speed > 0.0:
                v_next = v_next - outward_speed * radial
            b_next = b_next_clipped
        else:
            b_next = b_next_clipped

        d_next = d  # observer updates disturbance outside the prediction model
        return np.hstack([b_next, v_next, u_next, u_eff_next, d_next]).astype(float)

    def predict_bend(self, x: np.ndarray, dU: np.ndarray, settings: MpcSettings | None = None) -> np.ndarray:
        return self.step(x, dU, settings)[0:2]


class GrayboxObserver:
    """Velocity + bounded model-uncertainty observer for bend dynamics.

    The extra state d=[dx,dy] is an additive bend-acceleration disturbance.
    It is estimated from one-step prediction innovation and then fed back into
    the NMPC rollout. 
    """

    def __init__(self, model: AdaptiveResidualModel, settings: ObserverSettings):
        self.model = model
        self.settings = settings
        self.x: np.ndarray | None = None
        self.prev_bend: np.ndarray | None = None
        self.prev_time: float | None = None
        self.bend_dot_f = np.zeros(2, dtype=float)
        self.last_predicted_x: np.ndarray | None = None

        self.last_innovation = np.zeros(2, dtype=float)
        self.last_innovation_deg = 0.0
        self.last_raw_disturbance = np.zeros(2, dtype=float)
        self.last_disturbance_update_enabled = False
        self.last_saturation_active = False

    def initialize(self, bend: np.ndarray, tension_cmd: np.ndarray) -> np.ndarray:
        self.prev_bend = np.asarray(bend, dtype=float).reshape(2)
        self.bend_dot_f = np.zeros(2, dtype=float)
        self.last_innovation = np.zeros(2, dtype=float)
        self.last_raw_disturbance = np.zeros(2, dtype=float)
        self.last_innovation_deg = 0.0
        self.last_disturbance_update_enabled = False
        self.last_saturation_active = False
        self.x = self.model.make_state(bend, tension_cmd)
        return self.x.copy()

    def before_control_prediction(self, predicted_x: np.ndarray) -> None:
        self.last_predicted_x = np.asarray(predicted_x, dtype=float).reshape(self.model.nx).copy()

    def _saturation_active(self, u: np.ndarray, u_min: float | None, u_max: float | None) -> bool:
        """Freeze observer only for real actuator saturation.

        """
        if u_min is None or u_max is None:
            return False
        u = np.asarray(u, dtype=float).reshape(3)
        margin = max(float(self.settings.saturation_margin_N), 0.0)
        low_count = int(np.sum(u <= float(u_min) + margin))
        high_count = int(np.sum(u >= float(u_max) - margin))
        return bool(low_count >= 2 or high_count >= 1)

    def update(
        self,
        bend_meas: np.ndarray,
        tension_sent: np.ndarray,
        dt_meas: float | None = None,
        u_min: float | None = None,
        u_max: float | None = None,
    ) -> np.ndarray:
        if self.x is None:
            return self.initialize(bend_meas, tension_sent)

        b = np.asarray(bend_meas, dtype=float).reshape(2)
        u = np.asarray(tension_sent, dtype=float).reshape(3)
        dt = self.model.dt if dt_meas is None else max(float(dt_meas), 1e-6)

        if self.prev_bend is None:
            raw_dot = np.zeros(2, dtype=float)
        else:
            raw_dot = (b - self.prev_bend) / dt

        max_dot = np.deg2rad(float(self.settings.max_bend_dot_deg_s))
        raw_dot = clip_norm(raw_dot, max_dot)

        tau_v = max(float(self.settings.velocity_filter_tau), 1e-6)
        alpha_v = dt / (tau_v + dt)
        self.bend_dot_f = (1.0 - alpha_v) * self.bend_dot_f + alpha_v * raw_dot

        d_old = self.x[10:12].copy()
        d_new = float(self.settings.disturbance_decay) * d_old
        self.last_raw_disturbance = np.zeros(2, dtype=float)
        self.last_disturbance_update_enabled = False
        self.last_saturation_active = self._saturation_active(u, u_min, u_max)

        if self.last_predicted_x is not None:
            innovation = b - self.last_predicted_x[0:2]
            innovation_norm = float(np.linalg.norm(innovation))
            self.last_innovation_deg = float(np.rad2deg(innovation_norm))

            reset_jump = np.deg2rad(float(self.settings.reset_on_large_jump_deg))
            large_jump = innovation_norm > reset_jump

            deadband = np.deg2rad(float(self.settings.innovation_deadband_deg))
            if innovation_norm > deadband:
                innovation_eff = innovation * ((innovation_norm - deadband) / max(innovation_norm, 1e-12))
            else:
                innovation_eff = np.zeros(2, dtype=float)
            innovation_eff = clip_norm(innovation_eff, np.deg2rad(float(self.settings.innovation_clip_deg)))
            self.last_innovation = innovation_eff.copy()

            vel_innovation = self.bend_dot_f - self.last_predicted_x[2:4]
            vel_innovation = clip_norm(vel_innovation, max_dot)

            freeze = bool(self.settings.freeze_disturbance_when_saturated and self.last_saturation_active)
            if (not large_jump) and (not freeze):
                pos_corr = float(self.settings.disturbance_gain) * innovation_eff / max(self.model.dt ** 2, 1e-9)
                vel_corr = float(self.settings.disturbance_velocity_gain) * vel_innovation / max(self.model.dt, 1e-9)
                d_target = float(self.settings.disturbance_decay) * d_old + pos_corr + vel_corr

                tau_d = max(float(self.settings.disturbance_filter_tau), 1e-6)
                alpha_d = dt / (tau_d + dt)
                d_candidate = d_old + alpha_d * (d_target - d_old)
                max_step = float(self.settings.disturbance_rate_clip)
                if max_step > 0.0:
                    d_candidate = d_old + np.clip(d_candidate - d_old, -max_step, max_step)

                d_new = clip_norm(d_candidate, float(self.settings.disturbance_clip))
                self.last_raw_disturbance = (pos_corr + vel_corr).copy()
                self.last_disturbance_update_enabled = True
            elif large_jump:
                d_new = 0.8 * d_old
        else:
            self.last_innovation = np.zeros(2, dtype=float)
            self.last_innovation_deg = 0.0

        d_new = clip_norm(d_new, float(self.settings.disturbance_clip))

        u_eff = self.x[7:10] + self.model.alpha_u * (u - self.x[7:10])
        self.x = np.hstack([b, self.bend_dot_f, u, u_eff, d_new]).astype(float)
        self.prev_bend = b.copy()
        return self.x.copy()


# -----------------------------
# NMPC solver
# -----------------------------


def simulate_candidate(
    model: AdaptiveResidualModel,
    x0: np.ndarray,
    dU_flat: np.ndarray,
    target_seq: np.ndarray,
    settings: MpcSettings,
) -> float:
    dU = np.asarray(dU_flat, dtype=float).reshape(settings.horizon, 3)
    target_seq = np.asarray(target_seq, dtype=float)
    if target_seq.ndim == 1:
        target_seq = np.repeat(target_seq.reshape(1, 2), settings.horizon, axis=0)

    x = np.asarray(x0, dtype=float).reshape(model.nx).copy()
    cost = 0.0
    for i in range(settings.horizon):
        x = model.step(x, dU[i], settings)
        b = x[0:2]
        u_cmd = x[4:7]
        err = b - target_seq[i]
        cost += float(settings.qy) * float(err @ err)
        cost += float(settings.rdu) * float(dU[i] @ dU[i])
        if settings.rdd > 0.0:
            if i == 0:
                dd = dU[i]
            else:
                dd = dU[i] - dU[i - 1]
            cost += float(settings.rdd) * float(dd @ dd)
        cost += float(settings.ru) * float((u_cmd - settings.u_bias) @ (u_cmd - settings.u_bias))
        if settings.qtheta > 0:
            theta_ref = float(np.linalg.norm(target_seq[i]))
            theta = float(np.linalg.norm(b))
            cost += float(settings.qtheta) * (theta - theta_ref) ** 2
        if settings.qdist > 0:
            cost += float(settings.qdist) * float(x[10:12] @ x[10:12])

        max_bend = np.deg2rad(float(settings.max_bend_deg))
        excess = max(0.0, float(np.linalg.norm(b)) - max_bend)
        cost += 1e5 * excess * excess

        low_excess = np.maximum(0.0, float(settings.u_min) - x[7:10])
        high_excess = np.maximum(0.0, x[7:10] - float(settings.u_max))
        cost += 1e3 * float(low_excess @ low_excess + high_excess @ high_excess)

    terminal_err = x[0:2] - target_seq[-1]
    cost += float(settings.qf) * float(terminal_err @ terminal_err)
    return float(cost)


def solve_mpc(
    model: AdaptiveResidualModel,
    x0: np.ndarray,
    target_seq: np.ndarray,
    previous_dU: np.ndarray | None,
    settings: MpcSettings,
) -> tuple[np.ndarray, np.ndarray, bool, float, np.ndarray]:
    horizon = int(settings.horizon)
    if previous_dU is None or np.asarray(previous_dU).shape != (horizon, 3):
        dU0 = np.zeros((horizon, 3), dtype=float)
    else:
        previous_dU = np.asarray(previous_dU, dtype=float).reshape(horizon, 3)
        dU0 = np.vstack([previous_dU[1:], previous_dU[-1:]])

    bounds = [(-float(settings.max_du_step), float(settings.max_du_step))] * (horizon * 3)

    def obj(z: np.ndarray) -> float:
        return simulate_candidate(model, x0, z, target_seq, settings)

    res = minimize(
        obj,
        dU0.reshape(-1),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 15, "ftol": 1e-4, "maxls": 20},
    )

    if np.all(np.isfinite(res.x)):
        dU_seq = res.x.reshape(horizon, 3)
    else:
        print("[warn] optimizer returned invalid values; holding shifted previous solution")
        dU_seq = dU0

    x1_pred = model.step(np.asarray(x0, dtype=float), dU_seq[0], settings)
    u_cmd = x1_pred[4:7]
    cost = float(res.fun if np.isfinite(res.fun) else np.nan)
    return u_cmd, dU_seq, bool(res.success), cost, x1_pred


# -----------------------------
# Training utilities
# -----------------------------


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = X.shape[1]
    A = X.T @ X + float(lam) * np.eye(n)
    B = X.T @ Y
    return np.linalg.solve(A, B)


def sparsify_ridge(X: np.ndarray, Y: np.ndarray, lam: float = 1e-4, threshold: float = 1e-3, n_iter: int = 6) -> np.ndarray:
    W = ridge_fit(X, Y, lam)
    for _ in range(int(n_iter)):
        small = np.abs(W) < float(threshold)
        W[small] = 0.0
        for out in range(Y.shape[1]):
            keep = np.abs(W[:, out]) > 0.0
            if not np.any(keep):
                continue
            W_keep = ridge_fit(X[:, keep], Y[:, out:out+1], lam).reshape(-1)
            W[:, out] = 0.0
            W[keep, out] = W_keep
    return W
