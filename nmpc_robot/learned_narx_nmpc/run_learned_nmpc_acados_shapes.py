from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hardware_robot_friction import RobotHardware
from learned_nmpc_acados import (
    MpcSettings,
    append_keep,
    bend_to_phi_theta,
    circle_reference,
    fixed_reference,
    make_augmented_state,
    mpc_step_narx_acados,
    phi_theta_to_bend,
    setup_narx_acados_ocp_solver,
    smoothstep01,
)


def _load_params_module(module_or_path: str):
    """Load a params module by module name or by .py file path.

    Examples:
        --params params_learned_acados_friction_star
        --params params_learned_acados_friction_star.py
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
    default="params_learned_acados_friction_star",
    help=(
        "Params module name or .py path. Examples: "
        "params_learned_acados_friction_star, "
        "params_learned_acados_friction_triangle.py"
    ),
)
_ARGS, _UNKNOWN = _arg_parser.parse_known_args()
_PARAMS = _load_params_module(_ARGS.params)

acados = _PARAMS.acados
hardware = _PARAMS.hardware
model = _PARAMS.model
mpc = _PARAMS.mpc
run = _PARAMS.run
safety = _PARAMS.safety

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
            direction=str(run.get("direction", "cw")),
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
            direction=str(run.get("direction", "cw")),
            ramp_time=float(run["target_ramp_time"]),
            points=int(run.get("star_points", 5)),
            smooth_edges=bool(run.get("star_smooth_edges", True)),
        )

    raise ValueError("run['mode'] must be 'fixed', 'circle', 'triangle', or 'star'")


def build_bend_goal_sequence(t_now: float, dt: float, start_bend: np.ndarray, horizon: int) -> np.ndarray:
    """Return bend reference with shape (2, horizon+1)."""
    refs = [build_target(t_now + i * dt, start_bend) for i in range(horizon + 1)]
    return np.asarray(refs, dtype=float).T


def plot_tendon_rates(rows, out_dir: Path, dt: float):
    """Plot commanded and actual tendon rates, with the NMPC rate limit shown."""
    if not rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray([r["time_s"] for r in rows], dtype=float)
    max_rate = float(mpc["max_du_step"]) / float(dt)

    cmd_rate = np.asarray(
        [[r["dT1_rate_cmd_N_s"], r["dT2_rate_cmd_N_s"], r["dT3_rate_cmd_N_s"]] for r in rows],
        dtype=float,
    )
    actual_rate = np.asarray(
        [[r["dT1_rate_actual_N_s"], r["dT2_rate_actual_N_s"], r["dT3_rate_actual_N_s"]] for r in rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(3):
        ax.plot(t, cmd_rate[:, i], label=f"cmd dT{i + 1}/dt")
    ax.axhline(max_rate, linestyle="--", linewidth=1.0, label="+ rate limit")
    ax.axhline(-max_rate, linestyle="--", linewidth=1.0, label="- rate limit")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Commanded tendon rate [N/s]")
    ax.set_title("Optimizer-commanded tendon rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "tendon_rate_commanded.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(3):
        ax.plot(t, actual_rate[:, i], label=f"actual dT{i + 1}/dt")
    ax.axhline(max_rate, linestyle="--", linewidth=1.0, label="+ rate limit")
    ax.axhline(-max_rate, linestyle="--", linewidth=1.0, label="- rate limit")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Actual sent tendon rate [N/s]")
    ax.set_title("Actual sent tendon rate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "tendon_rate_actual.png", dpi=200)
    plt.close(fig)

    sat_threshold = 0.98 * float(mpc["max_du_step"])
    sat_counts = np.sum(np.abs(cmd_rate * dt) >= sat_threshold, axis=0)
    sat_percent = 100.0 * sat_counts / max(1, len(rows))
    print(
        "tendon-rate saturation percentage "
        f"[T1,T2,T3] = {np.round(sat_percent, 1)}% "
        f"using limit {mpc['max_du_step']} N/step = {max_rate:.3f} N/s"
    )


def main():
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
    out_dir = Path(run["out_dir"]) / f"acados_narx_nmpc_{run['mode']}_{timestamp}"

    print("============================================================")
    print("acados learned NARX-NMPC hardware run [shapes]")
    print("============================================================")
    print("params:", _ARGS.params)
    print("model:", model["path"])
    print("mode:", run["mode"])
    print("dt:", dt)
    print("duration:", run["duration"])
    print("tension bounds:", mpc["u_min"], mpc["u_max"])
    print("max_du_step:", mpc["max_du_step"], "N/step")
    print("horizon:", mpc["horizon"])
    print("output:", out_dir)
    print("============================================================")

    print("Building/loading acados solver before hardware initialization...")
    solver, learned_model = setup_narx_acados_ocp_solver(
        model["path"],
        mpc_settings,
        dt=dt,
        code_export_directory=acados.get("code_export_directory", "c_generated_code_narx_ocp"),
        solver_name=acados.get("solver_name", "narx_bend_ocp"),
        use_theta_constraint=bool(acados.get("use_theta_constraint", False)),
        use_model_clipping=bool(acados.get("use_model_clipping", False)),
        nlp_solver_type=str(acados.get("nlp_solver_type", "SQP")),
        nlp_solver_max_iter=int(acados.get("nlp_solver_max_iter", 20)),
        levenberg_marquardt=float(acados.get("levenberg_marquardt", 1e-2)),
        globalization=str(acados.get("globalization", "MERIT_BACKTRACKING")),
    )
    max_hist_len = learned_model.max_lag + 1
    print(
        f"NARX memory: ny={learned_model.ny}, nu={learned_model.nu}, "
        f"ndu={learned_model.ndu}, augmented nx={learned_model.nx}"
    )
    print("acados solver ready.")
    print("============================================================")

    if safety["confirm_before_start"]:
        input("Press Enter to initialize hardware and start... ")

    robot = RobotHardware(hardware, mpc)
    rows = []
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

        for k in tqdm(range(n_steps), desc="acados narx nmpc"):
            loop0 = time.perf_counter()
            u_prev = last_u.copy()
            t_now = k * dt

            bend_goal = build_bend_goal_sequence(t_now, dt, start_bend, mpc_settings.horizon)
            target_bend = bend_goal[:, 0]

            x0 = make_augmented_state(y_hist, u_hist, learned_model)

            solve0 = time.perf_counter()
            u_cmd, dU0, _x1_pred, status = mpc_step_narx_acados(
                solver,
                learned_model,
                x0,
                bend_goal,
                mpc_settings,
                return_status=True,
            )
            solve_time = time.perf_counter() - solve0
            opt_ok = status == 0

            if not opt_ok and k % 10 == 0:
                print(f"[warn] acados status={status}; holding previous tension for this step")

            u_sent, currents = robot.send_tensions(u_cmd)

            dT_actual = np.asarray(u_sent, dtype=float) - np.asarray(u_prev, dtype=float)
            dT_cmd_rate = np.asarray(dU0, dtype=float) / dt
            dT_actual_rate = dT_actual / dt

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

            try:
                opt_cost = float(solver.get_cost())
            except Exception:
                opt_cost = float("nan")

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
                "opt_ok": bool(opt_ok),
                "acados_status": int(status),
                "solve_time_s": float(solve_time),
                "opt_cost": opt_cost,
            }
            rows.append(row)

            if k % 10 == 0:
                print(
                    f"t={row['time_s']:.1f}s "
                    f"bend=[{row['bx_deg']:+.1f},{row['by_deg']:+.1f}] deg "
                    f"target=[{row['target_bx_deg']:+.1f},{row['target_by_deg']:+.1f}] deg "
                    f"err={row['bend_error_deg']:.1f} deg "
                    f"T={np.round(u_sent, 2)} "
                    f"dT={np.round(dU0, 3)} "
                    f"dT/dt={np.round(dT_cmd_rate, 2)} N/s "
                    f"status={status} "
                    f"solve={solve_time:.4f}s"
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
    plot_tendon_rates(rows, out_dir, dt)
    print_summary(rows)
    print("saved:", csv_path)
    print("plots:", out_dir)


if __name__ == "__main__":
    main()
