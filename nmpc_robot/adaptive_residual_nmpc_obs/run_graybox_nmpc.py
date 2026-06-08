from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from graybox_model import (
    AdaptiveResidualModel,
    GrayboxObserver,
    MpcSettings,
    ObserverSettings,
    bend_to_phi_theta,
    circle_reference,
    fixed_reference,
    phi_theta_to_bend,
    smoothstep01,
    solve_mpc,
)
from hardware_robot_graybox_friction import RobotHardware
from params_graybox import hardware, model, mpc, observer, run, safety
from save_results import print_summary, save_run


def ramp_to_tensions(robot, start_u, end_u, ramp_time, dt):
    n = max(1, int(round(float(ramp_time) / float(dt))))
    start_u = np.asarray(start_u, dtype=float)
    end_u = np.asarray(end_u, dtype=float)
    u_sent = start_u.copy()
    currents = np.zeros(3)
    for k in tqdm(range(n), desc="bias ramp"):
        loop0 = time.perf_counter()
        a = smoothstep01((k + 1) / n)
        u_des = (1.0 - a) * start_u + a * end_u
        u_sent, currents = robot.send_tensions(u_des)
        elapsed = time.perf_counter() - loop0
        if elapsed < dt:
            time.sleep(dt - elapsed)
    return u_sent, currents


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


def build_target_sequence(t_now: float, dt: float, start_bend: np.ndarray, horizon: int) -> np.ndarray:
    return np.asarray([build_target(t_now + (i + 1) * dt, start_bend) for i in range(horizon)], dtype=float)


def main():
    gray_model = AdaptiveResidualModel(model["path"])
    if "tendon_lag_tau" in model:
        gray_model.set_tendon_lag_tau(float(model["tendon_lag_tau"]))
    dt = float(run["dt"])
    if abs(gray_model.dt - dt) > 1e-9:
        print(f"[warn] run dt={dt} but model dt={gray_model.dt}. The model was trained with its own dt.")

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
        qtheta=float(mpc.get("qtheta", 0.0)),
        rdu=float(mpc["rdu"]),
        ru=float(mpc["ru"]),
        qdist=float(mpc.get("qdist", 0.0)),
        u_bias=initial_bias,
        max_bend_deg=float(safety["max_theta_deg"]),
        rdd=float(mpc.get("rdd", 0.0)),
    )

    obs_settings = ObserverSettings(
        velocity_filter_tau=float(observer.get("velocity_filter_tau", 0.12)),
        disturbance_gain=float(observer.get("disturbance_gain", 0.012)),
        disturbance_velocity_gain=float(observer.get("disturbance_velocity_gain", 0.004)),
        disturbance_decay=float(observer.get("disturbance_decay", 0.995)),
        disturbance_filter_tau=float(observer.get("disturbance_filter_tau", 0.8)),
        disturbance_clip=float(observer.get("disturbance_clip", 0.20)),
        disturbance_rate_clip=float(observer.get("disturbance_rate_clip", 0.025)),
        innovation_deadband_deg=float(observer.get("innovation_deadband_deg", 0.4)),
        innovation_clip_deg=float(observer.get("innovation_clip_deg", 8.0)),
        freeze_disturbance_when_saturated=bool(observer.get("freeze_disturbance_when_saturated", False)),
        saturation_margin_N=float(observer.get("saturation_margin_N", 0.03)),
        reset_on_large_jump_deg=float(observer.get("reset_on_large_jump_deg", 35.0)),
        max_bend_dot_deg_s=float(observer.get("max_bend_dot_deg_s", 300.0)),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(run["out_dir"]) / f"adaptive_residual_graybox_nmpc_{run['mode']}_{timestamp}"

    print("============================================================")
    print("Adaptive Residual Physics-Informed NMPC hardware run")
    print("============================================================")
    print("model:", model["path"])
    print("mode:", run["mode"])
    print("dt:", dt)
    print("duration:", run["duration"])
    print("tension bounds:", mpc["u_min"], mpc["u_max"])
    print("max_du_step:", mpc["max_du_step"], "N/step")
    print("horizon:", mpc["horizon"])
    print("model lag tau/alpha:", gray_model.tendon_lag_tau, gray_model.alpha_u)
    print("observer gain/vel_gain/decay/clip:",
          observer.get("disturbance_gain"),
          observer.get("disturbance_velocity_gain"),
          observer.get("disturbance_decay"),
          observer.get("disturbance_clip"))
    print("output:", out_dir)
    print("============================================================")

    if safety["confirm_before_start"]:
        input("Press Enter to initialize hardware and start... ")

    robot = RobotHardware(hardware, mpc)
    obs = GrayboxObserver(gray_model, obs_settings)
    rows = []
    previous_dU = None
    last_u = pretension_vec.copy()

    robot.init()

    try:
        phi, theta, phi_dot, theta_dot, bend = robot.read_state()
        x_est = obs.initialize(bend, pretension_vec)

        last_u, _ = ramp_to_tensions(
            robot,
            pretension_vec,
            initial_bias,
            float(mpc["bias_ramp_time"]),
            dt,
        )

        n_settle = max(1, int(round(float(mpc["settle_time"]) / dt)))
        for _ in range(n_settle):
            loop0 = time.perf_counter()
            u_sent, _ = robot.send_tensions(initial_bias)
            elapsed = time.perf_counter() - loop0
            if elapsed < dt:
                time.sleep(dt - elapsed)
            _, _, _, _, bend = robot.read_state()
            x_est = obs.update(bend, u_sent, dt_meas=dt, u_min=mpc_settings.u_min, u_max=mpc_settings.u_max)
            last_u = u_sent.copy()

        start_bend = x_est[0:2].copy()
        n_steps = max(1, int(round(float(run["duration"]) / dt)))
        t0 = time.perf_counter()

        last_obs_time = time.perf_counter()

        for k in tqdm(range(n_steps), desc="adaptive residual nmpc"):
            loop0 = time.perf_counter()
            t_now = k * dt

            target_bend = build_target(t_now, start_bend)
            target_seq = build_target_sequence(t_now, dt, start_bend, mpc_settings.horizon)

            solve0 = time.perf_counter()
            u_cmd, previous_dU, opt_ok, opt_cost, x1_pred = solve_mpc(
                gray_model,
                x_est,
                target_seq,
                previous_dU,
                mpc_settings,
            )
            solve_time = time.perf_counter() - solve0

            obs.before_control_prediction(x1_pred)
            u_sent, currents = robot.send_tensions(u_cmd)

            elapsed = time.perf_counter() - loop0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            phi, theta, phi_dot, theta_dot, bend = robot.read_state()

            now_obs = time.perf_counter()
            dt_obs = now_obs - last_obs_time
            last_obs_time = now_obs

            dt_obs = min(max(dt_obs, 1e-3), 0.20)

            x_est = obs.update(
                bend,
                u_sent,
                dt_meas=dt_obs,
                u_min=mpc_settings.u_min,
                u_max=mpc_settings.u_max,
            )

            last_u = u_sent.copy()

            phi_ref, theta_ref = bend_to_phi_theta(target_bend)
            bend_err_deg = float(np.rad2deg(np.linalg.norm(bend - target_bend)))
            theta_deg = float(np.rad2deg(theta))

            dU0 = previous_dU[0] if previous_dU is not None else np.zeros(3)
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
                "dT1_N": float(dU0[0]),
                "dT2_N": float(dU0[1]),
                "dT3_N": float(dU0[2]),
                "I1_mA": float(currents[0]),
                "I2_mA": float(currents[1]),
                "I3_mA": float(currents[2]),
                "vbx_rad_s": float(x_est[2]),
                "vby_rad_s": float(x_est[3]),
                "Teff1_N": float(x_est[7]),
                "Teff2_N": float(x_est[8]),
                "Teff3_N": float(x_est[9]),
                "dist_x_rad_s2": float(x_est[10]),
                "dist_y_rad_s2": float(x_est[11]),
                "dist_norm_rad_s2": float(np.linalg.norm(x_est[10:12])),
                "observer_innovation_deg": float(obs.last_innovation_deg),
                "observer_raw_dist_x_rad_s2": float(obs.last_raw_disturbance[0]),
                "observer_raw_dist_y_rad_s2": float(obs.last_raw_disturbance[1]),
                "observer_update_enabled": bool(obs.last_disturbance_update_enabled),
                "observer_saturation_active": bool(obs.last_saturation_active),
                "dt_obs_s": float(dt_obs),
                "opt_ok": bool(opt_ok),
                "opt_cost": float(opt_cost),
                "solve_time_s": float(solve_time),
            }
            rows.append(row)

            if k % 10 == 0:
                print(
                    f"t={row['time_s']:.1f}s "
                    f"bend=[{row['bx_deg']:+.1f},{row['by_deg']:+.1f}] deg "
                    f"target=[{row['target_bx_deg']:+.1f},{row['target_by_deg']:+.1f}] deg "
                    f"err={row['bend_error_deg']:.1f} deg "
                    f"T={np.round(u_sent, 2)} "
                    f"d={np.round(x_est[10:12], 2)} "
                    f"opt={opt_ok} solve={solve_time:.3f}s"
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
