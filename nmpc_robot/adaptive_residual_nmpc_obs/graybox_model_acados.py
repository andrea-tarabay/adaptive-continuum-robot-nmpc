from __future__ import annotations

"""acados solver wrapper for the adaptive residual gray-box NMPC.

"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import shutil

import numpy as np

from graybox_model import AdaptiveResidualModel, MpcSettings


@dataclass
class AcadosRuntimeInfo:
    status: int
    success: bool
    cost: float


def _require_acados_imports():
    """Import acados/casadi only when the acados solver is actually built."""
    try:
        import casadi as ca 
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver  
    except Exception as exc:  
        raise ImportError(
            "Could not import casadi/acados_template. Activate the environment "
            "where acados is installed, then run this script again. Original error: "
            f"{exc}"
        ) from exc
    return ca, AcadosModel, AcadosOcp, AcadosOcpSolver


def _as_np3(value: Any, default: float = 0.0) -> np.ndarray:
    if value is None:
        return np.full(3, float(default), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(3, float(arr), dtype=float)
    return arr.reshape(3)


def _add_acados_windows_dll_paths(code_dir: Path | None = None) -> None:
    """Help Windows find acados-generated and acados dependency DLLs.

    This mirrors the working learned_nmpc_acados.py behavior, but also adds the
    generated-code directory used by this gray-box solver.
    """
    candidate_dirs = []
    if code_dir is not None:
        candidate_dirs.append(Path(code_dir))

    for env_name in ("ACADOS_INSTALL_DIR", "ACADOS_SOURCE_DIR"):
        root = os.environ.get(env_name, "")
        if root:
            root_path = Path(root)
            candidate_dirs.extend([
                root_path,
                root_path / "bin",
                root_path / "lib",
                root_path / "build",
                root_path / "external" / "blasfeo" / "lib",
                root_path / "external" / "hpipm" / "lib",
            ])

    candidate_dirs.extend([
        Path(r"C:\Users\Admin\acados"),
        Path(r"C:\Users\Admin\acados\bin"),
        Path(r"C:\Users\Admin\acados\lib"),
        Path(r"C:\Users\Admin\acados\build"),
        Path(r"C:\lib"),
    ])

    for dll_dir in candidate_dirs:
        try:
            dll_dir = Path(dll_dir)
            if dll_dir.exists():
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(dll_dir))
                except Exception:
                    pass
        except Exception:
            pass


def _copy_windows_lib_prefixed_solver_if_needed(code_dir: Path, model_name: str) -> bool:
    """Windows sometimes builds libacados_ocp_solver_NAME.dll while acados
    tries to load acados_ocp_solver_NAME.dll. Copy the file to the expected name.
    """
    if os.name != "nt":
        return False

    code_dir = Path(code_dir)
    src = code_dir / f"libacados_ocp_solver_{model_name}.dll"
    dst = code_dir / f"acados_ocp_solver_{model_name}.dll"

    if src.exists() and not dst.exists():
        shutil.copyfile(src, dst)
        return True
    return dst.exists()


class AcadosGrayboxMpcSolver:
    """Fast acados implementation of solve_mpc(...).

    Usage inside the hardware loop:
        solver = AcadosGrayboxMpcSolver(gray_model, mpc_settings, acados_cfg)
        u_cmd, previous_dU, ok, cost, x1_pred = solver.solve(x_est, target_seq, previous_dU)
    """

    nx_model = 12
    nx_aug = 15
    nu = 3
    np_dim = 5  # [target_bx, target_by, u_bias1, u_bias2, u_bias3]

    def __init__(self, model: AdaptiveResidualModel, settings: MpcSettings, acados_cfg: dict | None = None):
        self.model_np = model
        self.settings = settings
        self.cfg = dict(acados_cfg or {})
        self.N = int(settings.horizon)
        self.dt = float(model.dt)
        self.last_info = AcadosRuntimeInfo(status=-999, success=False, cost=float("nan"))

        self.ca, AcadosModel, AcadosOcp, AcadosOcpSolver = _require_acados_imports()
        ocp = self._create_ocp(AcadosModel, AcadosOcp)

        json_file = str(self.cfg.get("json_file", "acados_ocp_graybox_residual.json"))
        generate = bool(self.cfg.get("generate", True))
        build = bool(self.cfg.get("build", True))
        verbose = bool(self.cfg.get("verbose", False))

        code_dir = Path(getattr(ocp, "code_export_directory", "c_generated_code")).resolve()
        _add_acados_windows_dll_paths(code_dir)

        def _construct_solver(generate_flag: bool, build_flag: bool):
            try:
                return AcadosOcpSolver(
                    ocp,
                    json_file=json_file,
                    generate=generate_flag,
                    build=build_flag,
                    verbose=verbose,
                )
            except TypeError:
                return AcadosOcpSolver(
                    ocp,
                    json_file=json_file,
                    generate=generate_flag,
                    build=build_flag,
                )

        try:
            self.solver = _construct_solver(generate, build)
        except FileNotFoundError:
            model_name = str(ocp.model.name)
            copied_or_present = _copy_windows_lib_prefixed_solver_if_needed(code_dir, model_name)
            if copied_or_present:
                _add_acados_windows_dll_paths(code_dir)
                self.solver = _construct_solver(False, False)
            else:
                raise

    # ------------------------------------------------------------------
    # CasADi model construction
    # ------------------------------------------------------------------

    def _tendon_moment_expr(self, teff):
        ca = self.ca
        pret = float(self.model_np.pretension)
        r = float(self.model_np.tendon_radius_m)
        angles = np.asarray(self.model_np.tendon_angles_rad, dtype=float).reshape(3)
        c = ca.DM(np.cos(angles))
        s = ca.DM(np.sin(angles))

        dT = teff - pret * ca.DM.ones(3, 1)
        dT = dT - ca.sum1(dT) / 3.0
        mx = r * ca.dot(c, dT)
        my = r * ca.dot(s, dT)
        return mx, my

    def _acceleration_expr(self, b, v, tcmd, teff, dU, disturbance):
        ca = self.ca
        mx, my = self._tendon_moment_expr(teff)
        bx, by = b[0], b[1]
        vx, vy = v[0], v[1]
        ue = teff - float(self.model_np.pretension) * ca.DM.ones(3, 1)

        f_phys = ca.vertcat(1.0, bx, by, vx, vy, mx, my)
        W_phys = ca.DM(np.asarray(self.model_np.W_phys, dtype=float))  # 7 x 2
        a_phys = ca.mtimes(W_phys.T, f_phys)

        f_res = ca.vertcat(
            1.0,
            bx, by, vx, vy, mx, my,
            dU[0], dU[1], dU[2],
            ue[0], ue[1], ue[2],
            ca.tanh(3.0 * vx), ca.tanh(3.0 * vy),
            bx * vx, by * vy,
            bx * mx, bx * my, by * mx, by * my,
            vx * mx, vx * my, vy * mx, vy * my,
            bx * bx, by * by, vx * vx, vy * vy, mx * mx, my * my, mx * my,
        )
        W_res = ca.DM(np.asarray(self.model_np.W_res, dtype=float))  # n_features x 2
        a_res = float(self.model_np.residual_scale) * ca.mtimes(W_res.T, f_res)

        a = a_phys + a_res + disturbance

        clip = float(self.model_np.accel_clip)
        if bool(self.cfg.get("smooth_accel_clip", True)) and clip > 0.0:
            a = clip * ca.tanh(a / clip)
        return a

    def _discrete_step_expr(self, x12, dU):
        ca = self.ca
        dt = float(self.model_np.dt)
        alpha = float(self.model_np.alpha_u)

        b = x12[0:2]
        v = x12[2:4]
        u_cmd = x12[4:7]
        u_eff = x12[7:10]
        disturbance = x12[10:12]

        u_next = u_cmd + dU
        u_eff_next = u_eff + alpha * (u_next - u_eff)
        a = self._acceleration_expr(b, v, u_next, u_eff_next, dU, disturbance)

        v_next = v + dt * a
        b_next = b + dt * v_next
        d_next = disturbance
        return ca.vertcat(b_next, v_next, u_next, u_eff_next, d_next)

    def _create_ocp(self, AcadosModel, AcadosOcp):
        ca = self.ca
        settings = self.settings

        xa = ca.SX.sym("xa", self.nx_aug)
        dU = ca.SX.sym("dU", self.nu)
        p = ca.SX.sym("p", self.np_dim)

        x12 = xa[0:self.nx_model]
        dU_prev = xa[self.nx_model:self.nx_aug]
        x12_next = self._discrete_step_expr(x12, dU)
        xa_next = ca.vertcat(x12_next, dU)

        target_b = p[0:2]
        u_bias = p[2:5]
        b_cost = x12_next[0:2]
        u_next = x12_next[4:7]
        dist_next = x12_next[10:12]

        theta_cost = ca.sqrt(ca.sumsqr(b_cost) + 1e-12)
        theta_ref = ca.sqrt(ca.sumsqr(target_b) + 1e-12)

        cost_y_expr = ca.vertcat(
            np.sqrt(float(settings.qy)) * (b_cost - target_b),
            np.sqrt(float(max(settings.qtheta, 0.0))) * (theta_cost - theta_ref),
            np.sqrt(float(settings.rdu)) * dU,
            np.sqrt(float(max(settings.rdd, 0.0))) * (dU - dU_prev),
            np.sqrt(float(settings.ru)) * (u_next - u_bias),
            np.sqrt(float(max(settings.qdist, 0.0))) * dist_next,
        )

        b_terminal = xa[0:2]
        dist_terminal = xa[10:12]
        theta_terminal = ca.sqrt(ca.sumsqr(b_terminal) + 1e-12)
        cost_y_expr_e = ca.vertcat(
            np.sqrt(float(settings.qf)) * (b_terminal - target_b),
            np.sqrt(float(max(settings.qtheta, 0.0))) * (theta_terminal - theta_ref),
            np.sqrt(float(max(settings.qdist, 0.0))) * dist_terminal,
        )

        model = AcadosModel()
        model.name = str(self.cfg.get("model_name", "graybox_residual_discrete"))
        model.x = xa
        model.u = dU
        model.p = p
        model.disc_dyn_expr = xa_next
        model.cost_y_expr = cost_y_expr
        model.cost_y_expr_e = cost_y_expr_e

        max_bend_rad = np.deg2rad(float(settings.max_bend_deg))
        model.con_h_expr = ca.vertcat(
            u_next,
            ca.sumsqr(b_cost),
        )
        model.con_h_expr_e = ca.vertcat(ca.sumsqr(b_terminal))

        ocp = AcadosOcp()
        ocp.model = model
        try:
            ocp.solver_options.N_horizon = int(settings.horizon)
        except Exception:
            ocp.dims.N = int(settings.horizon)
        ocp.parameter_values = np.zeros(self.np_dim)

        ny = int(cost_y_expr.shape[0])
        ny_e = int(cost_y_expr_e.shape[0])
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(ny)
        ocp.cost.W_e = np.eye(ny_e)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        ocp.constraints.x0 = np.zeros(self.nx_aug)

        ocp.constraints.idxbu = np.array([0, 1, 2], dtype=int)
        ocp.constraints.lbu = -float(settings.max_du_step) * np.ones(3)
        ocp.constraints.ubu = float(settings.max_du_step) * np.ones(3)

        ocp.constraints.lh = np.array([settings.u_min, settings.u_min, settings.u_min, 0.0], dtype=float)
        ocp.constraints.uh = np.array([settings.u_max, settings.u_max, settings.u_max, max_bend_rad ** 2], dtype=float)
        ocp.constraints.lh_e = np.array([0.0], dtype=float)
        ocp.constraints.uh_e = np.array([max_bend_rad ** 2], dtype=float)

        opts = ocp.solver_options
        opts.integrator_type = "DISCRETE"
        opts.tf = float(settings.horizon) * float(self.model_np.dt)
        opts.qp_solver = str(self.cfg.get("qp_solver", "PARTIAL_CONDENSING_HPIPM"))
        opts.hessian_approx = str(self.cfg.get("hessian_approx", "GAUSS_NEWTON"))
        opts.nlp_solver_type = str(self.cfg.get("nlp_solver_type", "SQP_RTI"))
        opts.nlp_solver_max_iter = int(self.cfg.get("nlp_solver_max_iter", 1))
        opts.qp_solver_iter_max = int(self.cfg.get("qp_solver_iter_max", 50))
        opts.print_level = int(self.cfg.get("print_level", 0))

        for name, value in {
            "regularize_method": self.cfg.get("regularize_method", "CONVEXIFY"),
            "qp_solver_warm_start": int(self.cfg.get("qp_solver_warm_start", 1)),
            "levenberg_marquardt": float(self.cfg.get("levenberg_marquardt", 1e-6)),
        }.items():
            try:
                setattr(opts, name, value)
            except Exception:
                pass

        project_dir = Path(__file__).resolve().parent
        code_dir = project_dir / str(self.cfg.get("code_export_directory", "c_generated_code"))
        ocp.code_export_directory = str(code_dir)

        return ocp

    # ------------------------------------------------------------------
    # Runtime solve
    # ------------------------------------------------------------------

    def _initial_augmented_state(self, x0: np.ndarray) -> np.ndarray:
        x0 = np.asarray(x0, dtype=float).reshape(self.nx_model)
        return np.hstack([x0, np.zeros(3, dtype=float)])

    def _shifted_guess(self, previous_dU: np.ndarray | None) -> np.ndarray:
        if previous_dU is None or np.asarray(previous_dU).shape != (self.N, 3):
            return np.zeros((self.N, 3), dtype=float)
        prev = np.asarray(previous_dU, dtype=float).reshape(self.N, 3)
        return np.vstack([prev[1:], prev[-1:]])

    def _set_parameters(self, target_seq: np.ndarray) -> None:
        target_seq = np.asarray(target_seq, dtype=float)
        if target_seq.ndim == 1:
            target_seq = np.repeat(target_seq.reshape(1, 2), self.N, axis=0)
        if target_seq.shape != (self.N, 2):
            raise ValueError(f"target_seq must have shape ({self.N}, 2), got {target_seq.shape}")

        u_bias = np.asarray(self.settings.u_bias, dtype=float).reshape(3)
        for i in range(self.N):
            p_i = np.hstack([target_seq[i], u_bias])
            self.solver.set(i, "p", p_i)
        self.solver.set(self.N, "p", np.hstack([target_seq[-1], u_bias]))

    def _warm_start(self, x0: np.ndarray, dU_guess: np.ndarray) -> None:
        xa = self._initial_augmented_state(x0)
        self.solver.set(0, "x", xa)
        self.solver.set(0, "lbx", xa)
        self.solver.set(0, "ubx", xa)

        x12 = np.asarray(x0, dtype=float).reshape(self.nx_model).copy()
        dprev = np.zeros(3, dtype=float)
        for i in range(self.N):
            du_i = np.asarray(dU_guess[i], dtype=float).reshape(3)
            self.solver.set(i, "u", du_i)
            if i > 0:
                self.solver.set(i, "x", np.hstack([x12, dprev]))
            x12 = self.model_np.step(x12, du_i, self.settings)
            dprev = du_i.copy()
        self.solver.set(self.N, "x", np.hstack([x12, dprev]))

    def solve(
        self,
        x0: np.ndarray,
        target_seq: np.ndarray,
        previous_dU: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, bool, float, np.ndarray]:
        x0 = np.asarray(x0, dtype=float).reshape(self.nx_model)
        dU_guess = self._shifted_guess(previous_dU)

        self._set_parameters(target_seq)
        self._warm_start(x0, dU_guess)

        status = int(self.solver.solve())
        success = status == 0

        if success:
            dU_seq = np.vstack([np.asarray(self.solver.get(i, "u"), dtype=float).reshape(1, 3) for i in range(self.N)])
            cost = self._safe_get_cost()
        else:
            dU_seq = dU_guess.copy()
            cost = float("nan")

        if not np.all(np.isfinite(dU_seq)):
            dU_seq = dU_guess.copy()
            success = False

        x1_pred = self.model_np.step(x0, dU_seq[0], self.settings)
        u_cmd = x1_pred[4:7].copy()
        self.last_info = AcadosRuntimeInfo(status=status, success=success, cost=cost)
        return u_cmd, dU_seq, success, cost, x1_pred

    def _safe_get_cost(self) -> float:
        try:
            return float(self.solver.get_cost())
        except Exception:
            return float("nan")


def solve_mpc_acados(
    solver: AcadosGrayboxMpcSolver,
    x0: np.ndarray,
    target_seq: np.ndarray,
    previous_dU: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, bool, float, np.ndarray]:
    """Small helper so the hardware loop looks like the old solve_mpc(...) call."""
    return solver.solve(x0, target_seq, previous_dU)
