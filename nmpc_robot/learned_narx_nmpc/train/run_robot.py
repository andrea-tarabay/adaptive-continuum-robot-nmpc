from __future__ import annotations

import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hardware_robot import RobotHardware
from learned_nmpc import (
    MpcSettings,
    NarxModel,
    append_keep,
    bend_to_phi_theta,
    circle_reference,
    fixed_reference,
    phi_theta_to_bend,
    smoothstep01,
    solve_mpc,
)
from params import hardware, model, mpc, run, safety
from save_results import print_summary, save_run


def ramp_to_tensions(robot, start_u, end_u, ramp_time, dt, y_hist, u_hist, max_hist_len):
    n = max(1, int(round(float(ramp_time) / float(dt))))
    start_u = np.asarray(start_u, dtype=float)
    end_u = np.asarray(end_u, dtype=float)

    for k in tqdm(range(n), desc="bias ramp"):
        loop0 = time.perf_counter()
        a = smoothstep01((k + 1) / n)
        u_des = (1.0 - a) * start_u + a * end_u
        u_sent, _ = robot.send_tensions(u_des)

        elapsed = time.perf_counter() - loop0
        if elapsed < dt:
            time.sleep(dt - elapsed)

        _, _, _, _, bend = robot.read_state()
        y_hist = append_keep(y_hist, bend, max_hist_len)
        u_hist = append_keep(u_hist, u_sent, max_hist_len)

    return y_hist, u_hist


def build_target(t, start_bend):
    if run["mode"] == "fixed":
        target = phi_theta_to_bend(
            np.deg2rad(float(run["target_phi_deg"])),
            np.deg2rad(float(run["target_theta_deg"])),
        )
        return fixed_reference(t, start_bend, target, float(run["target_ramp_time"]))

    if run["mode"] == "circle":
        return circle_reference(
            t=t,
            start_bend=start_bend,
            theta_rad=np.deg2rad(float(run["circle_theta_deg"])),
            phi_start_rad=np.deg2rad(float(run["phi_start_deg"])),
            loop_time=float(run["loop_time"]),
            direction=str(run["direction"]),
            ramp_time=float(run["target_ramp_time"]),
        )

    raise ValueError("run['mode'] must be 'fixed' or 'circle'")


def main():
    learned_model = NarxModel(model["path"])
    max_hist_len = learned_model.max_lag + 1

    dt = float(run["dt"])
    pretension_vec = np.full(3, float(mpc["pretension"]), dtype=float)
    initial_bias = np.clip(
        np.asarray(mpc["initial_bias"], dtype=float),
        float(mpc["u_min"]),
        float(mpc["u_max"]),
    )

    mpc_settings = MpcSettings(
        horizon=int(mpc["horizon"]),
        u_min=float(mpc["u_min"]),
        u_max=float(mpc["u_max"]),
        max_du_step=float(mpc["max_du_step"]),
        qy=float(mpc["qy"]),
        qf=float(mpc["qf"]),
        rdu=float(mpc["rdu"]),
        ru=float(mpc["ru"]),
        u_bias=initial_bias,
        max_bend_deg=float(safety["max_theta_deg"]),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(run["out_dir"]) / f"learned_nmpc_{run['mode']}_{timestamp}"

    print("============================================================")
    print("learned NARX-NMPC hardware run")
    print("============================================================")
    print("model:", model["path"])
    print(f"model memory: ny={learned_model.ny}, nu={learned_model.nu}, ndu={learned_model.ndu}")
    print("mode:", run["mode"])
    print("dt:", dt)
    print("duration:", run["duration"])
    print("tension bounds:", mpc["u_min"], mpc["u_max"])
    print("max_du_step:", mpc["max_du_step"], "N/step")
    print("output:", out_dir)
    print("============================================================")

    if safety["confirm_before_start"]:
        input("Press Enter to initialize hardware and start... ")

    robot = RobotHardware(hardware, mpc)
    rows = []
    previous_dU = None
    last_u = pretension_vec.copy()

    robot.init()

    try:
        phi, theta, phi_dot, theta_dot, bend = robot.read_state()
        y_hist = np.repeat(bend.reshape(1, 2), max_hist_len, axis=0)
        u_hist = np.repeat(pretension_vec.reshape(1, 3), max_hist_len, axis=0)

        y_hist, u_hist = ramp_to_tensions(
            robot,
            pretension_vec,
            initial_bias,
            float(mpc["bias_ramp_time"]),
            dt,
            y_hist,
            u_hist,
            max_hist_len,
        )
        last_u = initial_bias.copy()

        n_settle = max(1, int(round(float(mpc["settle_time"]) / dt)))
        for _ in range(n_settle):
            loop0 = time.perf_counter()
            u_sent, _ = robot.send_tensions(initial_bias)
            elapsed = time.perf_counter() - loop0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            phi, theta, phi_dot, theta_dot, bend = robot.read_state()
            y_hist = append_keep(y_hist, bend, max_hist_len)
            u_hist = append_keep(u_hist, u_sent, max_hist_len)
            last_u = u_sent.copy()

        start_bend = y_hist[-1].copy()
        n_steps = max(1, int(round(float(run["duration"]) / dt)))
        t0 = time.perf_counter()

        for k in tqdm(range(n_steps), desc="learned nmpc"):
            loop0 = time.perf_counter()
            t_now = k * dt
            target_bend = build_target(t_now, start_bend)

            u_cmd, previous_dU, opt_ok, opt_cost = solve_mpc(
                learned_model,
                y_hist,
                u_hist,
                target_bend,
                previous_dU,
                mpc_settings,
            )

            u_sent, currents = robot.send_tensions(u_cmd)

            elapsed = time.perf_counter() - loop0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            phi, theta, phi_dot, theta_dot, bend = robot.read_state()
            y_hist = append_keep(y_hist, bend, max_hist_len)
            u_hist = append_keep(u_hist, u_sent, max_hist_len)
            last_u = u_sent.copy()

            phi_ref, theta_ref = bend_to_phi_theta(target_bend)
            bend_err_deg = float(np.rad2deg(np.linalg.norm(bend - target_bend)))
            theta_deg = float(np.rad2deg(theta))

            row = {
                "time_s": time.perf_counter() - t0,
                "phi_rad": phi,
                "theta_rad": theta,
                "phi_dot_rad_s": phi_dot,
                "theta_dot_rad_s": theta_dot,
                "phi_deg": float(np.rad2deg(phi)),
                "theta_deg": theta_deg,
                "phi_ref_deg": float(np.rad2deg(phi_ref)),
                "theta_ref_deg": float(np.rad2deg(theta_ref)),
                "bx_rad": float(bend[0]),
                "by_rad": float(bend[1]),
                "bx_deg": float(np.rad2deg(bend[0])),
                "by_deg": float(np.rad2deg(bend[1])),
                "target_bx_rad": float(target_bend[0]),
                "target_by_rad": float(target_bend[1]),
                "target_bx_deg": float(np.rad2deg(target_bend[0])),
                "target_by_deg": float(np.rad2deg(target_bend[1])),
                "bend_error_deg": bend_err_deg,
                "T1_N": float(u_sent[0]),
                "T2_N": float(u_sent[1]),
                "T3_N": float(u_sent[2]),
                "I1_mA": float(currents[0]),
                "I2_mA": float(currents[1]),
                "I3_mA": float(currents[2]),
                "opt_ok": bool(opt_ok),
                "opt_cost": float(opt_cost),
            }
            rows.append(row)

            if k % 10 == 0:
                print(
                    f"t={row['time_s']:.1f}s "
                    f"bend=[{row['bx_deg']:+.1f},{row['by_deg']:+.1f}] deg "
                    f"target=[{row['target_bx_deg']:+.1f},{row['target_by_deg']:+.1f}] deg "
                    f"err={row['bend_error_deg']:.1f} deg "
                    f"T={np.round(u_sent, 2)} opt={opt_ok}"
                )

            if abs(theta_deg) > float(safety["max_theta_deg"]):
                print(f"[safety stop] theta={theta_deg:.2f} deg > {safety['max_theta_deg']:.2f} deg")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt: stopping.")

    finally:
        try:
            n_down = max(1, int(round(float(safety["ramp_down_time"]) / dt)))
            for j in range(n_down):
                a = smoothstep01((j + 1) / n_down)
                u = (1.0 - a) * last_u + a * pretension_vec
                robot.send_tensions(u)
                time.sleep(dt)
        except Exception as exc:
            print("[warn] ramp down failed:", exc)

        robot.stop()

    csv_path = save_run(rows, out_dir)
    print_summary(rows)
    print("saved:", csv_path)
    print("plots:", out_dir)


if __name__ == "__main__":
    main()
