import math
import numpy as np

from b3_tendon_mapping import (
    bend_command_from_bend_error,
    solve_tensions_single,
)
from b2_current_actuator import CurrentTensionMapper


# ==========================================================
# USER SETTINGS
# ==========================================================
# Desired bend
PHI_REF_DEG = 180.0
THETA_REF_DEG = 10.0

# Assume current measured pose is straight
PHI_MEAS_DEG = 0.0
THETA_MEAS_DEG = 0.0

# Identified motor bend directions
SIGMA_DEG = np.array([180.0, -45.0, 65.0], dtype=float)

# Pulley radius [m]
R_PULLEY_M = 0.009

# I[mA] = K_I_PER_T * T[N] + B_I_MA
K_I_PER_T = 72.3
B_I_MA = -6.0

# Motor tightening sign convention
TIGHTEN_SIGN = (-1.0, -1.0, -1.0)

# Current saturation
I_MAX_MA = 200.0

# Bend-vector gain
K_BEND = np.array([3.5, 3.5], dtype=float)

# Tension bias and limits
T_BIAS_N = 0.0
T_MIN_N = 0.0
T_MAX_N = 3.0


# ==========================================================
# Main
# ==========================================================
def main():
    phi_ref_rad = math.radians(PHI_REF_DEG)
    theta_ref_rad = math.radians(THETA_REF_DEG)

    phi_meas_rad = math.radians(PHI_MEAS_DEG)
    theta_meas_rad = math.radians(THETA_MEAS_DEG)

    sigma_rad = np.deg2rad(SIGMA_DEG)

    mapper = CurrentTensionMapper(
        r_pulley_m=R_PULLEY_M,
        tension_to_current_gain_mA_per_N=K_I_PER_T,
        tension_to_current_offset_mA=B_I_MA,
        tighten_sign=TIGHTEN_SIGN,
        i_max_mA=I_MAX_MA,
    )

    # ------------------------------------------------------
    # Block 1: desired bend -> bend command
    # ------------------------------------------------------
    u_cmd, e_u, u_ref, u_meas = bend_command_from_bend_error(
        phi_ref_rad=phi_ref_rad,
        theta_ref_rad=theta_ref_rad,
        phi_meas_rad=phi_meas_rad,
        theta_meas_rad=theta_meas_rad,
        k_bend=K_BEND,
        phi_valid=False,   # straight pose -> phi not reliable
    )

    # ------------------------------------------------------
    # Block 2: bend command -> tendon tensions
    # ------------------------------------------------------
    tensions_N = solve_tensions_single(
        u_bend_cmd=u_cmd,
        sigma_rad=sigma_rad,
        t_bias=T_BIAS_N,
        t_min=T_MIN_N,
        t_max=T_MAX_N,
    )

    # ------------------------------------------------------
    # Block 3: tendon tensions -> motor currents
    # ------------------------------------------------------
    current_cmd_signed_mA = mapper.tensions_to_signed_currents_mA(tensions_N)

    torque_motor_nm = np.array(
        [mapper.tension_to_torque_nm(t) for t in tensions_N],
        dtype=float,
    )

    current_cmd_mag_mA = np.abs(current_cmd_signed_mA)

    # ------------------------------------------------------
    # Print everything
    # ------------------------------------------------------
    print("=" * 72)
    print("DRY RUN: desired bend -> bend command -> tensions -> currents")
    print("=" * 72)

    print("Desired bend:")
    print(f"  phi_ref   = {PHI_REF_DEG:.2f} deg")
    print(f"  theta_ref = {THETA_REF_DEG:.2f} deg")
    print()

    print("Assumed measured/current pose:")
    print(f"  phi_meas   = {PHI_MEAS_DEG:.2f} deg")
    print(f"  theta_meas = {THETA_MEAS_DEG:.2f} deg")
    print()

    print("Bend command:")
    print(f"  u_ref  = [{u_ref[0]:.6f}, {u_ref[1]:.6f}]")
    print(f"  u_meas = [{u_meas[0]:.6f}, {u_meas[1]:.6f}]")
    print(f"  e_u    = [{e_u[0]:.6f}, {e_u[1]:.6f}]")
    print(f"  u_cmd  = [{u_cmd[0]:.6f}, {u_cmd[1]:.6f}]")
    print()

    print("Tendon tensions:")
    for i, t in enumerate(tensions_N, start=1):
        print(f"  T_{i} = {t:.6f} N")
    print()

    print("Motor torques from pulley:")
    for i, tau_m in enumerate(torque_motor_nm, start=1):
        print(f"  tau_motor_{i} = {tau_m:.6f} N.m")
    print()

    print("Motor current commands:")
    for i, (imag, isigned) in enumerate(zip(current_cmd_mag_mA, current_cmd_signed_mA), start=1):
        print(f"  I_{i}_mag    = {imag:.3f} mA")
        print(f"  I_{i}_signed = {isigned:.3f} mA")
    print()

    print("Vector summary:")
    print(f"  tensions_N            = {np.round(tensions_N, 6)}")
    print(f"  torque_motor_nm       = {np.round(torque_motor_nm, 6)}")
    print(f"  current_cmd_signed_mA = {np.round(current_cmd_signed_mA, 3)}")
    print("=" * 72)


if __name__ == "__main__":
    main()