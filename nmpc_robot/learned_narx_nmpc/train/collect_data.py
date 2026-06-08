from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

os.chdir(Path(__file__).resolve().parent)

from hardware_robot import RobotHardware
from params import data_collection, hardware, mpc


def make_prbs(n, dt, u_min, u_max, pretension=0.45, amp=2.55, hold_s=2.0, seed=1):
    rng = np.random.default_rng(seed)
    hold = max(1, int(round(hold_s / dt)))

    low = float(pretension)
    high = min(float(u_max), low + float(amp))
    mid_low = min(float(u_max), low + 0.35 * float(amp))
    mid = min(float(u_max), low + 0.60 * float(amp))
    mid_high = min(float(u_max), low + 0.80 * float(amp))

    candidates = []
    candidates.append([low, low, low])

    for i in range(3):
        cmd = [low, low, low]
        cmd[i] = high
        candidates.append(cmd)

    for low_idx in range(3):
        cmd = [mid, mid, mid]
        cmd[low_idx] = low
        candidates.append(cmd)

    for low_idx in range(3):
        active = [i for i in range(3) if i != low_idx]
        cmd = [low, low, low]
        cmd[active[0]] = mid_high
        cmd[active[1]] = mid_low
        candidates.append(cmd)

        cmd = [low, low, low]
        cmd[active[0]] = mid_low
        cmd[active[1]] = mid_high
        candidates.append(cmd)

    candidates = np.clip(np.asarray(candidates, dtype=float), u_min, u_max)
    blocks = int(np.ceil(n / hold))
    ids = rng.integers(0, len(candidates), size=blocks)

    for b in range(1, blocks):
        if ids[b] == ids[b - 1]:
            ids[b] = (ids[b] + rng.integers(1, len(candidates))) % len(candidates)

    u = np.zeros((n, 3), dtype=float)
    for b, cid in enumerate(ids):
        u[b * hold : min((b + 1) * hold, n)] = candidates[cid]
    return u


def make_sine(n, dt, u_min, u_max, pretension=0.45, amp=2.55):
    t = np.arange(n) * dt
    low = float(pretension)
    high = min(float(u_max), low + float(amp))
    mid = 0.5 * (low + high)
    radius = 0.5 * (high - low)

    freqs = [0.020, 0.033, 0.047]
    phases = [0.0, 2.1, 4.2]
    u = np.zeros((n, 3), dtype=float)
    for j in range(3):
        u[:, j] = mid + radius * np.sin(2.0 * np.pi * freqs[j] * t + phases[j])
    return np.clip(u, u_min, u_max)


def make_sweep(dt, u_min, u_max, pretension=0.45, levels=None, hold_s=2.0):
    if levels is None:
        levels = [0.45, 1.2, 2.0, 2.8, float(u_max)]

    levels = [float(np.clip(x, u_min, u_max)) for x in levels]
    commands = []

    for high in levels[1:]:
        commands += [
            [high, pretension, pretension],
            [pretension, high, pretension],
            [pretension, pretension, high],
        ]

    for high in levels[2:]:
        mid = 0.5 * (pretension + high)
        commands += [
            [mid, mid, pretension],
            [mid, pretension, mid],
            [pretension, mid, mid],
            [high, mid, pretension],
            [mid, high, pretension],
            [high, pretension, mid],
            [mid, pretension, high],
            [pretension, high, mid],
            [pretension, mid, high],
        ]

    commands.append([pretension, pretension, pretension])
    commands = np.clip(np.asarray(commands, dtype=float), u_min, u_max)

    hold = max(1, int(round(hold_s / dt)))
    return np.repeat(commands, hold, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", choices=["prbs", "sine", "sweep"], default="prbs")
    parser.add_argument("--duration", type=float, default=240.0)
    parser.add_argument("--dt", type=float, default=data_collection["dt"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pretension", type=float, default=data_collection["pretension"])
    parser.add_argument("--amp", type=float, default=2.55)
    parser.add_argument("--hold-s", type=float, default=data_collection["hold_s"])
    parser.add_argument("--max-tension", type=float, default=data_collection["max_tension"])
    parser.add_argument("--out-dir", type=str, default=data_collection["out_dir"])
    args = parser.parse_args()

    u_min = float(mpc["u_min"])
    u_max = min(float(mpc["u_max"]), float(args.max_tension))

    if args.pattern == "sweep":
        commands = make_sweep(args.dt, u_min, u_max, args.pretension, hold_s=args.hold_s)
    else:
        n = int(round(args.duration / args.dt))
        if args.pattern == "prbs":
            commands = make_prbs(n, args.dt, u_min, u_max, args.pretension, args.amp, args.hold_s, args.seed)
        else:
            commands = make_sine(n, args.dt, u_min, u_max, args.pretension, args.amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sysid_{args.pattern}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    print("command min:", np.round(commands.min(axis=0), 3))
    print("command max:", np.round(commands.max(axis=0), 3))
    print("output:", out_path)

    robot = RobotHardware(hardware, mpc)
    rows = []
    robot.init()
    t0 = time.perf_counter()

    try:
        for k in tqdm(range(len(commands)), desc=f"collect {args.pattern}"):
            loop0 = time.perf_counter()
            u_sent, currents = robot.send_tensions(commands[k])

            elapsed = time.perf_counter() - loop0
            if elapsed < args.dt:
                time.sleep(args.dt - elapsed)

            phi, theta, phi_dot, theta_dot, bend = robot.read_state()
            rows.append([
                time.perf_counter() - t0,
                phi, theta, phi_dot, theta_dot,
                bend[0], bend[1],
                u_sent[0], u_sent[1], u_sent[2],
                currents[0], currents[1], currents[2],
            ])

            if k % 20 == 0:
                print("T:", np.round(u_sent, 3), "theta deg:", round(float(np.rad2deg(theta)), 2))

    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        robot.stop()

    header = "time_s,phi_rad,theta_rad,phi_dot_rad_s,theta_dot_rad_s,bx_rad,by_rad,T1_N,T2_N,T3_N,I1_mA,I2_mA,I3_mA"
    np.savetxt(out_path, np.asarray(rows, dtype=float), delimiter=",", header=header, comments="")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
