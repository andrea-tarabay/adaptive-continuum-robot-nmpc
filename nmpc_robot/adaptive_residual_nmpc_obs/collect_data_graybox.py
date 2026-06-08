from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from graybox_model import smoothstep01
from hardware_robot_graybox import RobotHardware
from params_graybox import data_collection, hardware, mpc, model as model_cfg


def tendon_basis(angles_rad: np.ndarray) -> np.ndarray:
    """2x3 differential tension basis. Columns map tendon delta tension to [mx,my]."""
    return np.vstack([np.cos(angles_rad), np.sin(angles_rad)])


def differential_tensions_from_bend_vector(vec2: np.ndarray, amplitude_N: float, angles_rad: np.ndarray) -> np.ndarray:
    B = tendon_basis(angles_rad)
    delta = B.T @ np.linalg.solve(B @ B.T + 1e-9 * np.eye(2), np.asarray(vec2, dtype=float).reshape(2))
    delta = delta - np.mean(delta)
    norm = np.max(np.abs(delta))
    if norm > 1e-9:
        delta = delta / norm * float(amplitude_N)
    return delta


def safe_rate_step(u_current: np.ndarray, u_target: np.ndarray, max_du_step: float, u_min: float, u_max: float) -> np.ndarray:
    du = np.clip(np.asarray(u_target) - np.asarray(u_current), -float(max_du_step), float(max_du_step))
    return np.clip(np.asarray(u_current) + du, float(u_min), float(u_max))


def read_row(robot: RobotHardware, t0: float, u_sent: np.ndarray, currents: np.ndarray, segment: str, k: int) -> dict:
    phi, theta, phi_dot, theta_dot, bend = robot.read_state()
    return {
        "time_s": time.perf_counter() - t0,
        "segment": segment,
        "k": int(k),
        "phi_rad": float(phi),
        "theta_rad": float(theta),
        "phi_dot_rad_s": float(phi_dot),
        "theta_dot_rad_s": float(theta_dot),
        "phi_deg": float(np.rad2deg(phi)),
        "theta_deg": float(np.rad2deg(theta)),
        "bx_rad": float(bend[0]),
        "by_rad": float(bend[1]),
        "bx_deg": float(np.rad2deg(bend[0])),
        "by_deg": float(np.rad2deg(bend[1])),
        "T1_N": float(u_sent[0]),
        "T2_N": float(u_sent[1]),
        "T3_N": float(u_sent[2]),
        "I1_mA": float(currents[0]),
        "I2_mA": float(currents[1]),
        "I3_mA": float(currents[2]),
    }


def run_segment(robot: RobotHardware, name: str, command_seq: np.ndarray, dt: float, out_dir: Path) -> Path:
    rows = []
    t0 = time.perf_counter()
    u_sent = command_seq[0]
    currents = np.zeros(3)
    for k, u_des in enumerate(tqdm(command_seq, desc=name)):
        loop0 = time.perf_counter()
        u_sent, currents = robot.send_tensions(u_des)
        elapsed = time.perf_counter() - loop0
        if elapsed < dt:
            time.sleep(dt - elapsed)
        rows.append(read_row(robot, t0, u_sent, currents, name, k))

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    print("saved:", csv_path)
    return csv_path


def make_static_sequence(cfg: dict, angles_rad: np.ndarray) -> np.ndarray:
    dt = float(cfg["dt"])
    n_hold = max(1, int(round(float(cfg["static_hold_time"]) / dt)))
    pre = float(cfg["pretension"])
    u_min = float(cfg["u_min"])
    u_max = float(cfg["u_max"])
    amp = float(cfg["prbs_amplitude_N"])

    targets = [np.full(3, pre)]
    for j in range(3):
        u = np.full(3, pre)
        u[j] += amp
        targets.append(np.clip(u, u_min, u_max))
    for j in range(3):
        u = np.full(3, pre + 0.35 * amp)
        u[j] -= 0.35 * amp
        targets.append(np.clip(u, u_min, u_max))
    for phi in np.linspace(0, 2 * np.pi, 7, endpoint=False):
        delta = differential_tensions_from_bend_vector([np.cos(phi), np.sin(phi)], amp, angles_rad)
        targets.append(np.clip(np.full(3, pre) + delta, u_min, u_max))

    seq = []
    last = np.full(3, pre)
    ramp_steps = max(1, int(round(1.5 / dt)))
    for target in targets:
        for r in range(ramp_steps):
            a = smoothstep01((r + 1) / ramp_steps)
            seq.append((1.0 - a) * last + a * target)
        for _ in range(n_hold):
            seq.append(target.copy())
        last = target.copy()
    return np.asarray(seq, dtype=float)


def make_prbs_sequence(cfg: dict, angles_rad: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(int(cfg["seed"]))
    dt = float(cfg["dt"])
    n = max(1, int(round(float(cfg["prbs_duration"]) / dt)))
    hold_steps = max(1, int(cfg["prbs_hold_steps"]))
    pre = float(cfg["pretension"])
    amp = float(cfg["prbs_amplitude_N"])
    u_min = float(cfg["u_min"])
    u_max = float(cfg["u_max"])
    max_du = float(cfg["max_du_step"])

    u = np.full(3, pre)
    target = u.copy()
    seq = []
    for k in range(n):
        if k % hold_steps == 0:
            if rng.random() < 0.75:
                phi = rng.uniform(-np.pi, np.pi)
                delta = differential_tensions_from_bend_vector([np.cos(phi), np.sin(phi)], amp * rng.uniform(0.35, 1.0), angles_rad)
                common = rng.uniform(0.0, 0.25)
                target = np.clip(np.full(3, pre + common) + delta, u_min, u_max)
            else:
                target = np.clip(pre + rng.uniform(-0.15, amp, size=3), u_min, u_max)
        u = safe_rate_step(u, target, max_du, u_min, u_max)
        seq.append(u.copy())
    return np.asarray(seq, dtype=float)


def make_sine_sequence(cfg: dict, angles_rad: np.ndarray) -> np.ndarray:
    dt = float(cfg["dt"])
    n = max(1, int(round(float(cfg["sine_duration"]) / dt)))
    pre = float(cfg["pretension"])
    amp = float(cfg["prbs_amplitude_N"])
    u_min = float(cfg["u_min"])
    u_max = float(cfg["u_max"])
    seq = []
    t = np.arange(n) * dt
    freqs = [0.025, 0.04, 0.07]
    phases = np.deg2rad([0, 120, 240])
    for ti in t:
        delta = np.zeros(3)
        for j in range(3):
            delta[j] = amp * (0.45 * np.sin(2 * np.pi * freqs[0] * ti + phases[j]) + 0.25 * np.sin(2 * np.pi * freqs[1] * ti + 2 * phases[j]) + 0.15 * np.sin(2 * np.pi * freqs[2] * ti + phases[j]))
        delta = delta - np.mean(delta)
        seq.append(np.clip(np.full(3, pre + 0.15) + delta, u_min, u_max))
    return np.asarray(seq, dtype=float)


def make_circle_sequence(cfg: dict, angles_rad: np.ndarray) -> np.ndarray:
    dt = float(cfg["dt"])
    n = max(1, int(round(float(cfg["circle_duration"]) / dt)))
    pre = float(cfg["pretension"])
    amp = float(cfg["circle_amplitude_N"])
    period = float(cfg["circle_period_s"])
    u_min = float(cfg["u_min"])
    u_max = float(cfg["u_max"])
    seq = []
    for k in range(n):
        t = k * dt
        phi = 2 * np.pi * t / max(period, 1e-9) + 0.25 * np.sin(2 * np.pi * t / max(3 * period, 1e-9))
        delta = differential_tensions_from_bend_vector([np.cos(phi), np.sin(phi)], amp, angles_rad)
        seq.append(np.clip(np.full(3, pre + 0.15) + delta, u_min, u_max))
    return np.asarray(seq, dtype=float)


def ramp_down(robot: RobotHardware, last_u: np.ndarray, cfg: dict):
    dt = float(cfg["dt"])
    pre = float(cfg["pretension"])
    n = max(1, int(round(3.0 / dt)))
    for k in range(n):
        a = smoothstep01((k + 1) / n)
        u = (1.0 - a) * last_u + a * np.full(3, pre)
        robot.send_tensions(u)
        time.sleep(dt)


def main():
    cfg = dict(data_collection)
    angles_rad = np.deg2rad(np.asarray(model_cfg["tendon_angles_deg"], dtype=float))
    out_dir = Path(cfg["out_dir"]) / datetime.now().strftime("graybox_data_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    mpc_for_hw = dict(mpc)
    mpc_for_hw["pretension"] = float(cfg["pretension"])
    mpc_for_hw["u_min"] = float(cfg["u_min"])
    mpc_for_hw["u_max"] = float(cfg["u_max"])

    print("============================================================")
    print("Gray-box experimental data collection")
    print("Output:", out_dir)
    print("dt:", cfg["dt"])
    print("tension bounds:", cfg["u_min"], cfg["u_max"])
    print("Segments: static, prbs, sine, circle")
    print("============================================================")
    if cfg.get("confirm_before_start", True):
        input("Press Enter to initialize hardware and start data collection... ")

    robot = RobotHardware(hardware, mpc_for_hw)
    last_u = np.full(3, float(cfg["pretension"]))
    robot.init()

    try:
        segments = [
            ("static", make_static_sequence(cfg, angles_rad)),
            ("prbs", make_prbs_sequence(cfg, angles_rad)),
            ("sine", make_sine_sequence(cfg, angles_rad)),
            ("circle_tension", make_circle_sequence(cfg, angles_rad)),
        ]
        for name, seq in segments:
            last_u = seq[-1].copy()
            print(f"Starting segment {name}: {len(seq)} samples, {len(seq) * float(cfg['dt']):.1f}s")
            run_segment(robot, name, seq, float(cfg["dt"]), out_dir)
            ramp_down(robot, last_u, cfg)
            last_u = np.full(3, float(cfg["pretension"]))
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stopping collection.")
    finally:
        try:
            ramp_down(robot, last_u, cfg)
        except Exception as exc:
            print("[warn] ramp-down failed:", exc)
        robot.stop()

    print("Data collection done:", out_dir)


if __name__ == "__main__":
    main()
