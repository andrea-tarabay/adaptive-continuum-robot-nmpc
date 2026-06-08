import math
import numpy as np


def bend_vector(
    phi_rad: float,
    theta_rad: float,
    phi_valid: bool = True,
    theta_valid_thresh_rad: float = math.radians(3.0),
) -> np.ndarray:
    if (not phi_valid) or (abs(theta_rad) < theta_valid_thresh_rad):
        return np.zeros(2, dtype=float)

    return np.array(
        [
            theta_rad * math.cos(phi_rad),
            theta_rad * math.sin(phi_rad),
        ],
        dtype=float,
    )


def desired_bend_vector(phi_ref_rad: float, theta_ref_rad: float) -> np.ndarray:
    return np.array(
        [
            theta_ref_rad * math.cos(phi_ref_rad),
            theta_ref_rad * math.sin(phi_ref_rad),
        ],
        dtype=float,
    )


def bend_command_from_bend_error(
    phi_ref_rad: float,
    theta_ref_rad: float,
    phi_meas_rad: float,
    theta_meas_rad: float,
    k_bend,
    phi_valid: bool = True,
    theta_valid_thresh_rad: float = math.radians(3.0),
):
    u_ref = desired_bend_vector(phi_ref_rad, theta_ref_rad)
    u_meas = bend_vector(
        phi_meas_rad,
        theta_meas_rad,
        phi_valid=phi_valid,
        theta_valid_thresh_rad=theta_valid_thresh_rad,
    )

    e_u = u_ref - u_meas

    k_bend = np.asarray(k_bend, dtype=float)
    if k_bend.size == 1:
        u_cmd = float(k_bend) * e_u
    elif k_bend.size == 2:
        u_cmd = k_bend * e_u
    else:
        raise ValueError("k_bend must be scalar or length-2")

    return u_cmd, e_u, u_ref, u_meas


def solve_tensions_single(
    u_bend_cmd,
    sigma_rad,
    t_bias: float = 0.2,
    t_min: float = 0.0,
    t_max: float = 3.0,
) -> np.ndarray:
    """
    Map 2D bend command into 3 tendon tensions.
    sigma_rad comes from identified motor directions
    """
    sigma_rad = np.asarray(sigma_rad, dtype=float)
    u_bend_cmd = np.asarray(u_bend_cmd, dtype=float)

    if sigma_rad.shape[0] != 3:
        raise ValueError("sigma_rad must contain 3 tendon angles")

    D = np.column_stack((np.cos(sigma_rad), np.sin(sigma_rad)))
    tensions = float(t_bias) + D @ u_bend_cmd
    tensions = np.clip(tensions, t_min, t_max)
    return tensions