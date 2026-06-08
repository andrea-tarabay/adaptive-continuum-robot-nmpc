"""
ACADOS setup for the hardware NMPC.

Experiment A bend-space version:
    state  = [phi, theta, phi_dot, theta_dot, T1, T2, T3]
    input  = [dT1_dt, dT2_dt, dT3_dt]
    cost   = [theta*cos(phi), theta*sin(phi), phi_dot, theta_dot, T, dT_dt]
"""

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import casadi as ca
import numpy as np
import os
import shutil
from pathlib import Path


def export_pcc_acados_model(pcc_arm, name="pcc_arm_ocp"):
    nx_arm = 4 * pcc_arm.num_segments
    nu_tendon = 3 * pcc_arm.num_segments
    nx = nx_arm + nu_tendon

    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu_tendon)
    p_global = ca.SX.sym("p", nx_arm + pcc_arm.num_adaptive_params)

    x_arm = x[:nx_arm]
    tendon_tension = x[nx_arm:]

    arm_dot = pcc_arm.dynamics_func(x_arm, tendon_tension, p_global)
    tendon_dot = u

    model = AcadosModel()
    model.name = name
    model.x = x
    model.u = u
    model.p_global = p_global
    model.xdot = ca.SX.sym("xdot", nx)
    model.f_expl_expr = ca.vertcat(arm_dot, tendon_dot)
    model.f_impl_expr = model.xdot - model.f_expl_expr

    return model


def bend_output_from_state(x_arm, num_segments):
    bend_terms = []

    for i in range(num_segments):
        phi = x_arm[2 * i]
        theta = x_arm[2 * i + 1]
        bend_terms += [theta * ca.cos(phi), theta * ca.sin(phi)]

    bend_vec = ca.vertcat(*bend_terms)
    q_dot = x_arm[2 * num_segments:]

    return ca.vertcat(bend_vec, q_dot)


def setup_ocp_solver(pcc_arm, mpc_parameters, n_horizon, tf, acados_parameters=None):
    ocp = AcadosOcp()
    model = export_pcc_acados_model(pcc_arm)
    ocp.model = model

    if acados_parameters is None:
        acados_parameters = {}

    nx = model.x.size()[0]
    nu = model.u.size()[0]

    nx_arm = 4 * pcc_arm.num_segments
    nu_tendon = 3 * pcc_arm.num_segments
    tension_idx = np.arange(nx_arm, nx_arm + nu_tendon, dtype=int)

    ny_bend = 4 * pcc_arm.num_segments
    ny = ny_bend + nu_tendon + nu
    ny_e = ny_bend

    u_bound = mpc_parameters["u_bound"]
    v_bound = mpc_parameters.get("v_bound", [-0.5, 0.5])

    q = mpc_parameters["Q"]
    qf = mpc_parameters["Qf"]
    r = mpc_parameters["R"]
    r_rate = mpc_parameters.get("R_rate", 0.5 * np.eye(nu))

    # ocp.solver_options.N_horizon = n_horizon
    # ocp.solver_options.tf = tf
    # ocp.solver_options.nlp_solver_max_iter = 100
    # ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    # ocp.solver_options.levenberg_marquardt = 1e-2
    # ocp.solver_options.globalization_use_SOC = True
    # ocp.solver_options.integrator_type = "IRK"
    # ocp.solver_options.nlp_solver_tol_stat = 5e-4
    # ocp.solver_options.nlp_solver_tol_eq = 1e-6
    # ocp.solver_options.nlp_solver_tol_ineq = 1e-6
    # ocp.solver_options.nlp_solver_tol_comp = 1e-6

    ocp.solver_options.N_horizon = n_horizon
    ocp.solver_options.tf = tf

    ocp.solver_options.integrator_type = acados_parameters.get("integrator_type", "IRK")

    ocp.solver_options.nlp_solver_type = acados_parameters.get("nlp_solver_type", "SQP")
    ocp.solver_options.nlp_solver_max_iter = int(acados_parameters.get("nlp_solver_max_iter", 20))

    ocp.solver_options.qp_solver = acados_parameters.get(
        "qp_solver",
        "PARTIAL_CONDENSING_HPIPM",
    )
    ocp.solver_options.hessian_approx = acados_parameters.get(
        "hessian_approx",
        "GAUSS_NEWTON",
    )

    ocp.solver_options.globalization = acados_parameters.get(
        "globalization",
        "MERIT_BACKTRACKING",
    )
    ocp.solver_options.levenberg_marquardt = float(
        acados_parameters.get("levenberg_marquardt", 1e-2)
    )

    ocp.solver_options.nlp_solver_tol_stat = float(
        acados_parameters.get("nlp_solver_tol_stat", 1e-4)
    )
    ocp.solver_options.nlp_solver_tol_eq = float(
        acados_parameters.get("nlp_solver_tol_eq", 1e-6)
    )
    ocp.solver_options.nlp_solver_tol_ineq = float(
        acados_parameters.get("nlp_solver_tol_ineq", 1e-6)
    )
    ocp.solver_options.nlp_solver_tol_comp = float(
        acados_parameters.get("nlp_solver_tol_comp", 1e-6)
    )

    ocp.solver_options.print_level = int(acados_parameters.get("print_level", 0))

    if "qp_solver_iter_max" in acados_parameters:
        ocp.solver_options.qp_solver_iter_max = int(acados_parameters["qp_solver_iter_max"])

    if "qp_solver_warm_start" in acados_parameters:
        ocp.solver_options.qp_solver_warm_start = int(acados_parameters["qp_solver_warm_start"])

    if "regularize_method" in acados_parameters:
        ocp.solver_options.regularize_method = acados_parameters["regularize_method"]

    if "globalization_use_SOC" in acados_parameters:
        ocp.solver_options.globalization_use_SOC = bool(
            acados_parameters["globalization_use_SOC"]
        )


    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    x_arm = model.x[:nx_arm]
    tendon_tension = model.x[nx_arm:nx]
    bend_output = bend_output_from_state(x_arm, pcc_arm.num_segments)

    ocp.model.cost_y_expr = ca.vertcat(
        bend_output,
        tendon_tension,
        model.u,
    )

    ocp.model.cost_y_expr_e = bend_output

    ocp.cost.W = np.block([
        [q, np.zeros((ny_bend, nu_tendon)), np.zeros((ny_bend, nu))],
        [np.zeros((nu_tendon, ny_bend)), r, np.zeros((nu_tendon, nu))],
        [np.zeros((nu, ny_bend)), np.zeros((nu, nu_tendon)), r_rate],
    ])

    ocp.cost.W_e = qf
    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    ocp.constraints.lbu = v_bound[0] * np.ones(nu)
    ocp.constraints.ubu = v_bound[1] * np.ones(nu)
    ocp.constraints.idxbu = np.arange(nu, dtype=int)

    ocp.constraints.idxbx = tension_idx
    ocp.constraints.lbx = u_bound[0] * np.ones(nu_tendon)
    ocp.constraints.ubx = u_bound[1] * np.ones(nu_tendon)

    ocp.constraints.idxbx_e = tension_idx
    ocp.constraints.lbx_e = u_bound[0] * np.ones(nu_tendon)
    ocp.constraints.ubx_e = u_bound[1] * np.ones(nu_tendon)

    ocp.constraints.x0 = np.zeros(nx)
    ocp.p_global_values = np.zeros(nx_arm + pcc_arm.num_adaptive_params)

    project_dir = Path(__file__).resolve().parent
    code_dir = project_dir / "c_generated_code_pcc_ocp"
    ocp.code_export_directory = str(code_dir)

    for dll_dir in [
        Path(r"C:\Users\Admin\acados\bin"),
        Path(r"C:\Users\Admin\acados\lib"),
    ]:
        if dll_dir.exists():
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(str(dll_dir))
            except Exception:
                pass

    try:
        solver = AcadosOcpSolver(ocp)
    except FileNotFoundError as e:
        src = code_dir / f"libacados_ocp_solver_{model.name}.dll"
        dst = code_dir / f"acados_ocp_solver_{model.name}.dll"

        if src.exists():
            shutil.copyfile(src, dst)
            solver = AcadosOcpSolver(ocp, generate=False, build=False)
        else:
            raise e

    return solver


def mpc_step_acados(
    ocp_solver,
    x0,
    bend_goal,
    p_adaptive,
    n_horizon,
    u_bound,
    tension_ref=None,
    return_status=False,
):
    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu
    nx_arm = nx - nu

    ocp_solver.set(0, "lbx", x0)
    ocp_solver.set(0, "ubx", x0)

    ocp_solver.set_p_global_and_precompute_dependencies(
        np.hstack([x0[:nx_arm], p_adaptive])
    )

    if tension_ref is None:
        t_ref = float(u_bound[0]) * np.ones(nu)
    else:
        t_ref = np.asarray(tension_ref, dtype=float).reshape(nu)
        t_ref = np.clip(t_ref, float(u_bound[0]), float(u_bound[1]))

    v_ref = np.zeros(nu)

    for i in range(n_horizon):
        yref_i = np.hstack([bend_goal[:, i], t_ref, v_ref])
        ocp_solver.set(i, "yref", yref_i)

    ocp_solver.set(n_horizon, "yref", bend_goal[:, n_horizon])

    status = ocp_solver.solve()

    if status != 0:
        print(f"Acados returned status {status}. Holding tensions.")
        v0 = np.zeros(nu)
        x1 = x0.copy()
        if return_status:
            return v0, x1, status
        return v0, x1

    v0 = ocp_solver.get(0, "u")
    x1 = ocp_solver.get(1, "x")

    if return_status:
        return v0, x1, status

    return v0, x1
