from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
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
)
from graybox_model_acados import AcadosGrayboxMpcSolver, solve_mpc_acados
from hardware_robot_graybox_friction import RobotHardware



def _load_params_module(module_or_path: str):
    """Load a params module by module name or by .py file path.

    Examples:
        --params params_graybox_acados_friction_triangle_graybox
        --params params_graybox_acados_friction_star_graybox.py
    """
    module_or_path = str(module_or_path)

    if module_or_path.endswith(".py") or "/" in module_or_path or "\\" in module_or_path:
        path = Path(module_or_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Could not find params file: {path}")

        module_name = path.stem.replace(" ", "_").replace("-", "_")
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load params file: {path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(module_or_path)


_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument(
    "--params",
    default="params_graybox_acados_friction_triangle_graybox",
    help=(
        "Params module name or .py path. Examples: "
        "params_graybox_acados_friction_triangle_graybox, "
        "params_graybox_acados_friction_star_graybox.py"
    ),
)
_ARGS, _UNKNOWN = _arg_parser.parse_known_args()
_PARAMS = _load_params_module(_ARGS.params)

acados = _PARAMS.acados
hardware = _PARAMS.hardware
model = _PARAMS.model
mpc = _PARAMS.mpc
observer = _PARAMS.observer
run = _PARAMS.run
safety = _PARAMS.safety
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


def _closed_polyline_reference(
    t: float,
    start_bend: np.ndarray,
    vertices: list[np.ndarray],
    loop_time: float,
    ramp_time: float,
    smooth_edges: bool = True,
) -> np.ndarray:
   
    start_bend = np.asarray(start_bend, dtype=float).reshape(2)
    vertices = [np.asarray(v, dtype=float).reshape(2) for v in vertices]
    if len(vertices) < 2:
        raise ValueError("Need at least two vertices for a closed polyline reference.")

    first = vertices[0]
    if t < ramp_time:
        a = smoothstep01(float(t) / max(float(ramp_time), 1e-9))
        return (1.0 - a) * start_bend + a * first

    tau = (float(t) - max(float(ramp_time), 0.0)) % max(float(loop_time), 1e-9)
    n_edges = len(vertices)
    edge_time = max(float(loop_time), 1e-9) / float(n_edges)
    edge_idx = int(np.floor(tau / edge_time))
    edge_idx = min(edge_idx, n_edges - 1)

    local = (tau - edge_idx * edge_time) / max(edge_time, 1e-9)
    if smooth_edges:
        local = smoothstep01(local)

    p0 = vertices[edge_idx]
    p1 = vertices[(edge_idx + 1) % n_edges]
    return (1.0 - local) * p0 + local * p1


def triangle_reference(
    t: float,
    start_bend: np.ndarray,
    theta_rad: float,
    phi_start_rad: float,
    loop_time: float,
    direction: str,
    ramp_time: float,
    smooth_edges: bool = True,
) -> np.ndarray:
    sign = -1.0 if str(direction).lower() == "cw" else 1.0
    vertices = [
        phi_theta_to_bend(phi_start_rad + sign * 2.0 * np.pi * i / 3.0, theta_rad)
        for i in range(3)
    ]
    return _closed_polyline_reference(
        t=t,
        start_bend=start_bend,
        vertices=vertices,
        loop_time=loop_time,
        ramp_time=ramp_time,
        smooth_edges=smooth_edges,
    )


def star_reference(
    t: float,
    start_bend: np.ndarray,
    outer_theta_rad: float,
    inner_theta_rad: float,
    phi_start_rad: float,
    loop_time: float,
    direction: str,
    ramp_time: float,
    points: int = 5,
    smooth_edges: bool = True,
) -> np.ndarray:
    """Five-point star outline in bend space.

    The path alternates outer and inner vertices:
        outer -> inner -> outer -> inner ...
    """
    points = max(3, int(points))
    sign = -1.0 if str(direction).lower() == "cw" else 1.0

    vertices = []
    for i in range(2 * points):
        theta = outer_theta_rad if i % 2 == 0 else inner_theta_rad
        phi = phi_start_rad + sign * np.pi * i / float(points)
        vertices.append(phi_theta_to_bend(phi, theta))

    return _closed_polyline_reference(
        t=t,
        start_bend=start_bend,
        vertices=vertices,
        loop_time=loop_time,
        ramp_time=ramp_time,
        smooth_edges=smooth_edges,
    )


def build_target(t, start_bend):
    mode = str(run["mode"]).lower()

    if mode == "fixed":
        target = phi_theta_to_bend(
            np.deg2rad(float(run["target_phi_deg"])),
            np.deg2rad(float(run["target_theta_deg"])),
        )
        return fixed_reference(t, start_bend, target, float(run["target_ramp_time"]))

    if mode == "circle":
        return circle_reference(
            t=t,
            start_bend=start_bend,
            theta_rad=np.deg2rad(float(run["circle_theta_deg"])),
            phi_start_rad=np.deg2rad(float(run["phi_start_deg"])),
            loop_time=float(run["loop_time"]),
            direction=str(run["direction"]),
            ramp_time=float(run["target_ramp_time"]),
        )

    if mode == "triangle":
        return triangle_reference(
            t=t,
            start_bend=start_bend,
            theta_rad=np.deg2rad(float(run["triangle_theta_deg"])),
            phi_start_rad=np.deg2rad(float(run["phi_start_deg"])),
            loop_time=float(run["loop_time"]),
            direction=str(run["direction"]),
            ramp_time=float(run["target_ramp_time"]),
            smooth_edges=bool(run.get("triangle_smooth_edges", True)),
        )

    if mode == "star":
        outer_theta = np.deg2rad(float(run["star_outer_theta_deg"]))
        if "star_inner_theta_deg" in run:
            inner_theta = np.deg2rad(float(run["star_inner_theta_deg"]))
        else:
            inner_theta = outer_theta * float(run.get("star_inner_ratio", 0.45))

        return star_reference(
            t=t,
            start_bend=start_bend,
            outer_theta_rad=outer_theta,
            inner_theta_rad=inner_theta,
            phi_start_rad=np.deg2rad(float(run["phi_start_deg"])),
            loop_time=float(run["loop_time"]),
            direction=str(run["direction"]),
            ramp_time=float(run["target_ramp_time"]),
            points=int(run.get("star_points", 5)),
            smooth_edges=bool(run.get("star_smooth_edges", True)),
        )

    raise ValueError("run['mode'] must be 'fixed', 'circle', 'triangle', or 'star'")


def build_target_sequence(t_now: float, dt: float, start_bend: np.ndarray, horizon: int) -> np.ndarray:
    return np.asarray([build_target(t_now + (i + 1) * dt, start_bend) for i in range(horizon)], dtype=float)


def main():
    gray_model = AdaptiveResidualModel(model["path"])
    if "tendon_lag_tau" in model:
        gray_model.set_tendon_lag_tau(float(model["tendon_lag_tau"]))

    dt = float(run["dt"])
    if abs(gray_model.dt - dt) > 1e-9:
        print(f"[warn] run dt={dt} but model dt={gray_model.dt}. acados uses model dt={gray_model.dt} internally.")
        print("       Best practice: train/identify the model with the same dt you use on hardware.")

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

    print("Building/loading acados solver...")
    acados_solver = AcadosGrayboxMpcSolver(gray_model, mpc_settings, acados)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(run["out_dir"]) / f"adaptive_residual_graybox_acados_{run['mode']}_{timestamp}"

    print("============================================================")
    print("Adaptive Residual Physics-Informed NMPC hardware run [acados]")
    print("============================================================")
    print("model:", model["path"])
    print("mode:", run["mode"])
    print("run dt:", dt)
    print("model dt:", gray_model.dt)
    print("duration:", run["duration"])
    print("tension bounds:", mpc["u_min"], mpc["u_max"])
    print("max_du_step:", mpc["max_du_step"], "N/step")
    print("horizon:", mpc["horizon"])
    print("acados solver:", acados.get("nlp_solver_type", "SQP_RTI"))
    print("model lag tau/alpha:", gray_model.tendon_lag_tau, gray_model.alpha_u)
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

        for k in tqdm(range(n_steps), desc="adaptive residual acados nmpc"):
            loop0 = time.perf_counter()
            u_prev = last_u.copy()  
            t_now = k * dt
            target_bend = build_target(t_now, start_bend)
            target_seq = build_target_sequence(t_now, dt, start_bend, mpc_settings.horizon)

            solve0 = time.perf_counter()
            u_cmd, previous_dU, opt_ok, opt_cost, x1_pred = solve_mpc_acados(
                acados_solver,
                x_est,
                target_seq,
                previous_dU,
            )
            solve_time = time.perf_counter() - solve0

            obs.before_control_prediction(x1_pred)
            u_sent, currents = robot.send_tensions(u_cmd)

            dU0 = previous_dU[0] if previous_dU is not None else np.zeros(3)
            dT_actual = np.asarray(u_sent, dtype=float) - np.asarray(u_prev, dtype=float)
            dT_cmd_rate = np.asarray(dU0, dtype=float) / dt
            dT_actual_rate = dT_actual / dt

            elapsed = time.perf_counter() - loop0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            phi, theta, phi_dot, theta_dot, bend = robot.read_state()
            x_est = obs.update(bend, u_sent, dt_meas=dt, u_min=mpc_settings.u_min, u_max=mpc_settings.u_max)
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
                "dT1_N": float(dU0[0]),
                "dT2_N": float(dU0[1]),
                "dT3_N": float(dU0[2]),
                "dT1_actual_N": float(dT_actual[0]),
                "dT2_actual_N": float(dT_actual[1]),
                "dT3_actual_N": float(dT_actual[2]),

                "dT1_rate_cmd_N_s": float(dT_cmd_rate[0]),
                "dT2_rate_cmd_N_s": float(dT_cmd_rate[1]),
                "dT3_rate_cmd_N_s": float(dT_cmd_rate[2]),

                "dT1_rate_actual_N_s": float(dT_actual_rate[0]),
                "dT2_rate_actual_N_s": float(dT_actual_rate[1]),
                "dT3_rate_actual_N_s": float(dT_actual_rate[2]),
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
                "opt_ok": bool(opt_ok),
                "acados_status": int(acados_solver.last_info.status),
                "opt_cost": float(opt_cost),
                "solve_time_s": float(solve_time),
            }

            for attr, prefix in [
                ("last_nominal_currents_mA", "I_nom"),
                ("last_friction_currents_mA", "I_fric"),
                ("last_sent_currents_mA", "I_sent"),
                ("last_motor_velocity_rad_s", "motor_vel"),
                ("last_measured_current_mA", "I_meas"),
                ("last_pwm_percent", "pwm"),
            ]:
                if hasattr(robot, attr):
                    vals = np.asarray(getattr(robot, attr), dtype=float).reshape(-1)
                    for j, val in enumerate(vals[:3], start=1):
                        row[f"{prefix}{j}"] = float(val)

            rows.append(row)

            if k % 10 == 0:
                print(
                    f"t={row['time_s']:.1f}s "
                    f"bend=[{row['bx_deg']:+.1f},{row['by_deg']:+.1f}] deg "
                    f"target=[{row['target_bx_deg']:+.1f},{row['target_by_deg']:+.1f}] deg "
                    f"err={row['bend_error_deg']:.1f} deg "
                    f"T={np.round(u_sent, 2)} "
                    f"d={np.round(x_est[10:12], 2)} "
                    f"opt={opt_ok} status={row['acados_status']} solve={solve_time:.4f}s"
                )

            if solve_time > 0.8 * dt:
                print(f"[timing warn] solve_time={solve_time:.4f}s is close to dt={dt:.4f}s")

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
