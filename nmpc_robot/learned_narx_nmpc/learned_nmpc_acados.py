"""
learned_nmpc_acados.py

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import shutil

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


def smoothstep01(a: float) -> float:
    a = np.clip(float(a), 0.0, 1.0)
    return float(a * a * (3.0 - 2.0 * a))


def wrap_angle_rad(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def phi_theta_to_bend(phi: float, theta: float) -> np.ndarray:
    return np.array([theta * np.cos(phi), theta * np.sin(phi)], dtype=float)


def bend_to_phi_theta(b: np.ndarray) -> tuple[float, float]:
    bx, by = float(b[0]), float(b[1])
    theta = float(np.hypot(bx, by))
    phi = 0.0 if theta < 1e-9 else float(np.arctan2(by, bx))
    return phi, theta




def fixed_reference(t: float, start_bend: np.ndarray, target_bend: np.ndarray, ramp_time: float) -> np.ndarray:
    a = smoothstep01(float(t) / max(float(ramp_time), 1e-9))
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
        a = smoothstep01(float(t) / max(float(ramp_time), 1e-9))
        return (1.0 - a) * np.asarray(start_bend, dtype=float) + a * first

    sign = -1.0 if str(direction).lower() == "cw" else 1.0
    tau = float(t) - max(float(ramp_time), 0.0)
    phi = float(phi_start_rad) + sign * 2.0 * np.pi * tau / max(float(loop_time), 1e-9)
    return phi_theta_to_bend(phi, theta_rad)


def append_keep(arr: np.ndarray, row: np.ndarray, max_len: int) -> np.ndarray:
    arr = np.vstack([arr, np.asarray(row, dtype=float).reshape(1, -1)])
    if len(arr) > max_len:
        arr = arr[-max_len:]
    return arr


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


class NarxAcadosModel:
    """Container for the learned NARX weights and dimensions."""

    def __init__(self, model_path: str | Path):
        self.path = str(model_path)
        data = np.load(self.path, allow_pickle=True)

        self.w = np.asarray(data["W"], dtype=float)
        self.mean = np.asarray(data["feature_mean"], dtype=float).reshape(-1)
        self.scale = np.asarray(data["feature_scale"], dtype=float).reshape(-1)

        self.ny = int(data["ny"])
        self.nu = int(data["nu"])
        self.ndu = int(data["ndu"])

        self.delta_clip = np.asarray(data["delta_clip"], dtype=float).reshape(-1)
        self.max_lag = max(self.ny, self.nu, self.ndu + 1)

        self.y_dim = 2
        self.u_dim = 3
        self.nx = self.y_dim * (self.ny + 1) + self.u_dim * (self.nu + 1)
        self.nctrl = self.u_dim

        expected_features = (
            1
            + self.y_dim * (self.ny + 1)
            + self.y_dim * self.ny
            + self.u_dim * (self.nu + 1)
            + self.u_dim * (self.ndu + 1)
        )
        if self.mean.size != expected_features:
            raise ValueError(
                f"Feature size mismatch. Expected {expected_features}, "
                f"but feature_mean has size {self.mean.size}."
            )
        if self.w.shape[0] != expected_features:
            raise ValueError(
                f"W has wrong number of rows. Expected {expected_features}, got {self.w.shape[0]}."
            )
        if self.w.shape[1] != self.y_dim:
            raise ValueError(f"W should have 2 output columns for [bx, by], got {self.w.shape[1]}.")

        if self.ndu + 1 > self.nu:
            raise ValueError(
                "This acados implementation computes du-lags from the u-history. "
                "It requires ndu + 1 <= nu."
            )

    @property
    def u_start(self) -> int:
        return self.y_dim * (self.ny + 1)

    @property
    def current_tension_slice(self) -> slice:
        return slice(self.u_start, self.u_start + self.u_dim)


def make_augmented_state(y_hist: np.ndarray, u_hist: np.ndarray, model: NarxAcadosModel) -> np.ndarray:
    """Build x0 from chronological histories.

    """
    y_hist = np.asarray(y_hist, dtype=float)
    u_hist = np.asarray(u_hist, dtype=float)

    need = model.max_lag + 1
    if y_hist.shape[0] < need or u_hist.shape[0] < need:
        raise ValueError(f"Need at least {need} history rows for this model.")

    ys = [y_hist[-1 - j].reshape(model.y_dim) for j in range(model.ny + 1)]
    us = [u_hist[-1 - j].reshape(model.u_dim) for j in range(model.nu + 1)]
    return np.hstack(ys + us).astype(float)


def current_tension_from_state(x: np.ndarray, model: NarxAcadosModel) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return x[model.current_tension_slice].copy()


def _slice_vec(x: ca.SX, start: int, length: int) -> ca.SX:
    return x[start : start + length]


def _casadi_narx_feature(x: ca.SX, model_data: NarxAcadosModel) -> ca.SX:
    """Build the exact same feature vector as make_feature(), but symbolically."""
    ny = model_data.ny
    nu = model_data.nu
    ndu = model_data.ndu
    y_dim = model_data.y_dim
    u_dim = model_data.u_dim

    y_start = 0
    u_start = model_data.u_start

    y_lags = [
        _slice_vec(x, y_start + y_dim * j, y_dim)
        for j in range(ny + 1)
    ]
    u_lags = [
        _slice_vec(x, u_start + u_dim * j, u_dim)
        for j in range(nu + 1)
    ]

    feat = [ca.SX(1.0)]

    feat.extend(y_lags)

    for lag in range(1, ny + 1):
        feat.append(y_lags[lag - 1] - y_lags[lag])

    feat.extend(u_lags)

    for lag in range(ndu + 1):
        feat.append(u_lags[lag] - u_lags[lag + 1])

    return ca.vertcat(*feat)


def export_narx_acados_model(
    learned_model: NarxAcadosModel,
    *,
    name: str = "narx_bend_ocp",
    use_model_clipping: bool = False,
) -> AcadosModel:
    
    x = ca.SX.sym("x", learned_model.nx)
    dU = ca.SX.sym("dU", learned_model.nctrl)

    y_current = x[: learned_model.y_dim]
    u_current = x[learned_model.current_tension_slice]

    feature = _casadi_narx_feature(x, learned_model)
    mean = ca.DM(learned_model.mean)
    scale = ca.DM(learned_model.scale)
    W = ca.DM(learned_model.w)

    feature_n = (feature - mean) / scale
    delta = W.T @ feature_n

    if use_model_clipping:
        clip = ca.DM(learned_model.delta_clip)
        delta = ca.fmin(ca.fmax(delta, -clip), clip)

    y_next = y_current + delta

    if use_model_clipping:
        y_next = ca.fmin(ca.fmax(y_next, -2.0), 2.0)

    u_next = u_current + dU

    old_y_without_oldest = x[0 : learned_model.y_dim * learned_model.ny]
    old_u_without_oldest = x[
        learned_model.u_start : learned_model.u_start + learned_model.u_dim * learned_model.nu
    ]

    x_next = ca.vertcat(y_next, old_y_without_oldest, u_next, old_u_without_oldest)

    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = dU
    model.disc_dyn_expr = x_next
    return model


def _add_acados_windows_dll_paths() -> None:
    """Help Windows find acados DLLs when running from PowerShell."""
    candidate_dirs = [
        Path(os.environ.get("ACADOS_INSTALL_DIR", "")) / "bin",
        Path(os.environ.get("ACADOS_INSTALL_DIR", "")) / "lib",
        Path(r"C:\Users\Admin\acados\bin"),
        Path(r"C:\Users\Admin\acados\lib"),
    ]
    for dll_dir in candidate_dirs:
        try:
            if dll_dir.exists():
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(dll_dir))
                except Exception:
                    pass
        except Exception:
            pass


def setup_narx_acados_ocp_solver(
    model_path: str | Path,
    settings: MpcSettings,
    *,
    dt: float,
    code_export_directory: str | Path = "c_generated_code_narx_ocp",
    solver_name: str = "narx_bend_ocp",
    use_theta_constraint: bool = False,
    use_model_clipping: bool = False,
    nlp_solver_type: str = "SQP",
    nlp_solver_max_iter: int = 20,
    levenberg_marquardt: float = 1e-2,
    globalization: str = "MERIT_BACKTRACKING",
) -> tuple[AcadosOcpSolver, NarxAcadosModel]:
    """Build and return the acados OCP solver for the learned NARX model."""
    learned_model = NarxAcadosModel(model_path)
    acados_model = export_narx_acados_model(
        learned_model,
        name=solver_name,
        use_model_clipping=use_model_clipping,
    )

    ocp = AcadosOcp()
    ocp.model = acados_model

    nx = learned_model.nx
    nu = learned_model.nctrl
    n_horizon = int(settings.horizon)

    ocp.solver_options.N_horizon = n_horizon
    ocp.solver_options.tf = float(n_horizon * dt)
    ocp.solver_options.integrator_type = "DISCRETE"

    ocp.solver_options.nlp_solver_type = str(nlp_solver_type)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_max_iter = int(nlp_solver_max_iter)
    ocp.solver_options.globalization = str(globalization)
    ocp.solver_options.levenberg_marquardt = float(levenberg_marquardt)
    ocp.solver_options.nlp_solver_tol_stat = 1e-4
    ocp.solver_options.nlp_solver_tol_eq = 1e-6
    ocp.solver_options.nlp_solver_tol_ineq = 1e-6
    ocp.solver_options.nlp_solver_tol_comp = 1e-6
    ocp.solver_options.print_level = 0
    current_tension = acados_model.x[learned_model.current_tension_slice]
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    bend_radius = ca.sqrt(acados_model.x[0] ** 2 + acados_model.x[1] ** 2 + 1e-9)

    ocp.model.cost_y_expr = ca.vertcat(
        acados_model.x[:2],      # bx, by
        bend_radius,             # theta/radius
        acados_model.u,          # dT
        current_tension,         # T
    )

    ocp.model.cost_y_expr_e = ca.vertcat(
        acados_model.x[:2],
        bend_radius,
    )

    qtheta = 220.0
    qf_theta = 800.0

    ocp.cost.W = np.diag(
        [
            float(settings.qy),
            float(settings.qy),
            qtheta,
            float(settings.rdu),
            float(settings.rdu),
            float(settings.rdu),
            float(settings.ru),
            float(settings.ru),
            float(settings.ru),
        ]
    )

    ocp.cost.W_e = np.diag(
        [
            float(settings.qf),
            float(settings.qf),
            qf_theta,
        ]
    )


    ocp.cost.yref = np.hstack([
    np.zeros(2),   # bx, by
    0.0,           # theta
    np.zeros(3),   # dT
    np.asarray(settings.u_bias, dtype=float).reshape(3),
])

    ocp.cost.yref_e = np.zeros(3)

    # Control bounds: dT per MPC step.
    ocp.constraints.idxbu = np.arange(nu, dtype=int)
    ocp.constraints.lbu = -float(settings.max_du_step) * np.ones(nu)
    ocp.constraints.ubu = float(settings.max_du_step) * np.ones(nu)

    # Tension bounds on the current tension part of the state.
    tension_idx = np.arange(learned_model.u_start, learned_model.u_start + learned_model.u_dim, dtype=int)
    ocp.constraints.idxbx = tension_idx
    ocp.constraints.lbx = float(settings.u_min) * np.ones(learned_model.u_dim)
    ocp.constraints.ubx = float(settings.u_max) * np.ones(learned_model.u_dim)

    ocp.constraints.idxbx_e = tension_idx
    ocp.constraints.lbx_e = float(settings.u_min) * np.ones(learned_model.u_dim)
    ocp.constraints.ubx_e = float(settings.u_max) * np.ones(learned_model.u_dim)

    # Initial state placeholder. The real x0 is set online every control step.
    ocp.constraints.x0 = np.zeros(nx)

    if use_theta_constraint:
        max_bend_rad = np.deg2rad(float(settings.max_bend_deg))
        bend_norm_sq = acados_model.x[0] ** 2 + acados_model.x[1] ** 2
        ocp.model.con_h_expr = ca.vertcat(bend_norm_sq)
        ocp.constraints.lh = np.array([0.0])
        ocp.constraints.uh = np.array([max_bend_rad**2])

        ocp.model.con_h_expr_e = ca.vertcat(bend_norm_sq)
        ocp.constraints.lh_e = np.array([0.0])
        ocp.constraints.uh_e = np.array([max_bend_rad**2])

    project_dir = Path(__file__).resolve().parent
    code_dir = project_dir / str(code_export_directory)
    ocp.code_export_directory = str(code_dir)

    _add_acados_windows_dll_paths()

    try:
        solver = AcadosOcpSolver(ocp)
    except FileNotFoundError as exc:
        src = code_dir / f"libacados_ocp_solver_{acados_model.name}.dll"
        dst = code_dir / f"acados_ocp_solver_{acados_model.name}.dll"
        if src.exists():
            shutil.copyfile(src, dst)
            solver = AcadosOcpSolver(ocp, generate=False, build=False)
        else:
            raise exc

    return solver, learned_model


def mpc_step_narx_acados(
    ocp_solver: AcadosOcpSolver,
    learned_model: NarxAcadosModel,
    x0: np.ndarray,
    bend_goal: np.ndarray,
    settings: MpcSettings,
    *,
    return_status: bool = False,
):
    n_horizon = int(settings.horizon)
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    bend_goal = np.asarray(bend_goal, dtype=float)

    if bend_goal.shape != (2, n_horizon + 1):
        raise ValueError(f"bend_goal must have shape (2, {n_horizon + 1}), got {bend_goal.shape}.")

    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)

    u_bias = np.asarray(settings.u_bias, dtype=float).reshape(3)

    for i in range(n_horizon):
        theta_ref_i = float(np.linalg.norm(bend_goal[:, i]))
        yref_i = np.hstack([
            bend_goal[:, i],
            theta_ref_i,
            np.zeros(3),
            u_bias,
        ])
        ocp_solver.set(i, "yref", yref_i)

    theta_ref_e = float(np.linalg.norm(bend_goal[:, n_horizon]))
    ocp_solver.set(
        n_horizon,
        "yref",
        np.hstack([bend_goal[:, n_horizon], theta_ref_e])
    )
    status = int(ocp_solver.solve())

    if status != 0:
        # Safe fallback: hold the current tension command.
        try:
            ocp_solver.reset()
        except Exception:
            pass
        dU0 = np.zeros(3, dtype=float)
        u_cmd = current_tension_from_state(x0, learned_model)
        x1_pred = x0.copy()
    else:
        dU0 = np.asarray(ocp_solver.get(0, "u"), dtype=float).reshape(3)
        x1_pred = np.asarray(ocp_solver.get(1, "x"), dtype=float).reshape(-1)

        u_cmd = current_tension_from_state(x0, learned_model) + dU0
        u_cmd = np.clip(u_cmd, float(settings.u_min), float(settings.u_max))

    if return_status:
        return u_cmd, dU0, x1_pred, status
    return u_cmd, dU0, x1_pred
