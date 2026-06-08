from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def smoothstep01(a: float) -> float:
    a = np.clip(float(a), 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def wrap_angle_rad(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def phi_theta_to_bend(phi: float, theta: float) -> np.ndarray:
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


def make_feature(y_hist: np.ndarray, u_hist: np.ndarray, k: int, ny: int, nu: int, ndu: int) -> np.ndarray:
    du = np.zeros_like(u_hist)
    du[1:] = u_hist[1:] - u_hist[:-1]

    dy = np.zeros_like(y_hist)
    dy[1:k + 1] = y_hist[1:k + 1] - y_hist[:k]

    feat = [1.0]

    for lag in range(ny + 1):
        feat.extend(y_hist[k - lag])

    for lag in range(1, ny + 1):
        feat.extend(dy[k - lag + 1])

    for lag in range(nu + 1):
        feat.extend(u_hist[k - lag])

    for lag in range(ndu + 1):
        feat.extend(du[k - lag])

    return np.asarray(feat, dtype=float).reshape(1, -1)


class NarxModel:
    def __init__(self, model_path: str):
        data = np.load(model_path, allow_pickle=True)
        self.w = data["W"]
        self.mean = data["feature_mean"]
        self.scale = data["feature_scale"]
        self.ny = int(data["ny"])
        self.nu = int(data["nu"])
        self.ndu = int(data["ndu"])
        self.delta_clip = np.asarray(data["delta_clip"], dtype=float)
        self.max_lag = max(self.ny, self.nu, self.ndu + 1)

    def predict_next(self, y_hist: np.ndarray, u_hist: np.ndarray) -> np.ndarray:
        k = len(y_hist) - 1
        x = make_feature(y_hist, u_hist, k, self.ny, self.nu, self.ndu)
        xn = (x - self.mean) / self.scale
        delta = (xn @ self.w).reshape(-1)
        delta = np.clip(delta, -self.delta_clip, self.delta_clip)
        return np.clip(y_hist[-1] + delta, -2.0, 2.0)


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


def simulate_candidate(
    model: NarxModel,
    y_hist0: np.ndarray,
    u_hist0: np.ndarray,
    dU_flat: np.ndarray,
    target: np.ndarray,
    settings: MpcSettings,
) -> float:
    dU = dU_flat.reshape(settings.horizon, 3)
    y_hist = y_hist0.copy()
    u_hist = u_hist0.copy()

    cost = 0.0
    for i in range(settings.horizon):
        u_next = np.clip(u_hist[-1] + dU[i], settings.u_min, settings.u_max)
        u_hist = np.vstack([u_hist, u_next])
        y_next = model.predict_next(y_hist, u_hist)
        y_hist = np.vstack([y_hist, y_next])

        err = y_next - target
        cost += settings.qy * float(err @ err)
        cost += settings.rdu * float(dU[i] @ dU[i])
        cost += settings.ru * float((u_next - settings.u_bias) @ (u_next - settings.u_bias))

    terminal_err = y_hist[-1] - target
    cost += settings.qf * float(terminal_err @ terminal_err)

    bend_norm = np.linalg.norm(y_hist[-settings.horizon:], axis=1)
    excess = np.maximum(0.0, bend_norm - np.deg2rad(settings.max_bend_deg))
    cost += 1e4 * float(np.sum(excess**2))
    return cost


def solve_mpc(
    model: NarxModel,
    y_hist: np.ndarray,
    u_hist: np.ndarray,
    target: np.ndarray,
    previous_dU: np.ndarray | None,
    settings: MpcSettings,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    horizon = settings.horizon

    if previous_dU is None or previous_dU.shape != (horizon, 3):
        dU0 = np.zeros((horizon, 3), dtype=float)
    else:
        dU0 = np.vstack([previous_dU[1:], previous_dU[-1:]])

    bounds = [(-settings.max_du_step, settings.max_du_step)] * (horizon * 3)

    def obj(x: np.ndarray) -> float:
        return simulate_candidate(model, y_hist, u_hist, x, target, settings)

    res = minimize(
        obj,
        dU0.reshape(-1),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 25, "ftol": 1e-4},
    )

    if res.success:
        dU_seq = res.x.reshape(horizon, 3)
    else:
        print("[warn] optimizer:", res.message)
        dU_seq = dU0

    u_cmd = np.clip(u_hist[-1] + dU_seq[0], settings.u_min, settings.u_max)
    cost = float(res.fun if np.isfinite(res.fun) else np.nan)
    return u_cmd, dU_seq, bool(res.success), cost


def fixed_reference(t: float, start_bend: np.ndarray, target_bend: np.ndarray, ramp_time: float) -> np.ndarray:
    a = smoothstep01(t / max(float(ramp_time), 1e-9))
    return (1.0 - a) * start_bend + a * target_bend


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
        return (1.0 - a) * start_bend + a * first

    sign = -1.0 if direction == "cw" else 1.0
    tau = t - max(float(ramp_time), 0.0)
    phi = phi_start_rad + sign * 2.0 * np.pi * tau / max(float(loop_time), 1e-9)
    return phi_theta_to_bend(phi, theta_rad)
