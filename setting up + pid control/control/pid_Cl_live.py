import math
import time

import numpy as np
import matplotlib.pyplot as plt

from b1_phi_theta import IMUBendReader
from b3_tendon_mapping import bend_command_from_bend_error, solve_tensions_single
from b2_current_actuator import CurrentTensionMapper
from dynamixel_controller import DynamixelController, BaseModel, PortCommError


# =========================================================
# USER SETTINGS
# =========================================================
IMU_PORT = "COM4"
DXL_PORT = "COM3"

MOTOR_IDS = [1, 2, 3]
DXL_BAUDRATE = 57600
DXL_PROTOCOL = 2.0
REVERSE_DIRECTION = False

# Desired bend target
PHI_REF_DEG = 180.0
THETA_REF_DEG = 8.0
SIGMA_DEG = np.array([180.0, -45.0, 70.0], dtype=float)

# Current / tension mapping
R_PULLEY_M = 0.009
A_MA_PER_N = 64.7
B_MA = 0.0
TIGHTEN_SIGN = (-1.0, -1.0, -1.0)

K_BEND = np.array([4, 4], dtype=float)     
KI_BEND = np.array([0, 0], dtype=float)    
KD_BEND = np.array([0, 0], dtype=float)   

INT_LIM = np.array([1.5, 1.5], dtype=float)
D_FILT_ALPHA = 0.25   
# Tension limits
T_BIAS_N = 0.15
T_MIN_N = 0.0
T_MAX_N = 3.0

# Current safety
I_MAX_MA = 120.0

# Angle safety
THETA_HARD_LIMIT_DEG = 35.0
PHI_VALID_THRESH_DEG = 3.0
PHI_OFFSET_DEG = 0.0

# Loop timing
DT = 0.02
PRINT_DT = 0.10
PLOT_DT = 0.05

# Live plot
ENABLE_LIVE_PLOT = True
PLOT_WINDOW_SEC = 20.0


# =========================================================
# Helpers
# =========================================================
def clamp_vec(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def safe_zero_currents(dxl, n_motors):
    try:
        dxl.set_goal_current_mA([0.0] * n_motors)
    except Exception:
        pass


def wrap_to_360(angle_deg):
    return float(angle_deg) % 360.0


# =========================================================
# Live plot helpers
# =========================================================
def init_live_plot():
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    line_theta, = ax[0].plot([], [], label="theta")
    line_theta_ref, = ax[0].plot([], [], "--", label="theta_ref")
    ax[0].set_ylabel("Theta [deg]")
    ax[0].grid(True)
    ax[0].legend(loc="upper right")

    line_phi, = ax[1].plot([], [], label="phi")
    line_phi_ref, = ax[1].plot([], [], "--", label="phi_ref")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Phi [deg]")
    ax[1].grid(True)
    ax[1].legend(loc="upper right")

    fig.suptitle("Closed-loop bend tracking")
    fig.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "line_theta": line_theta,
        "line_theta_ref": line_theta_ref,
        "line_phi": line_phi,
        "line_phi_ref": line_phi_ref,
        "t_hist": [],
        "theta_hist": [],
        "theta_ref_hist": [],
        "phi_hist": [],
        "phi_ref_hist": [],
        "plot_t0": time.time(),
    }


def update_live_plot(plot_handles, theta_deg, phi_deg, phi_valid, theta_ref_deg, phi_ref_deg):
    if plot_handles is None:
        return False

    fig = plot_handles["fig"]
    if not plt.fignum_exists(fig.number):
        return False

    t_now = time.time() - plot_handles["plot_t0"]

    plot_handles["t_hist"].append(t_now)
    plot_handles["theta_hist"].append(theta_deg)
    plot_handles["theta_ref_hist"].append(theta_ref_deg)
    plot_handles["phi_hist"].append(wrap_to_360(phi_deg) if phi_valid else np.nan)
    plot_handles["phi_ref_hist"].append(wrap_to_360(phi_ref_deg))

    while plot_handles["t_hist"] and (t_now - plot_handles["t_hist"][0]) > PLOT_WINDOW_SEC:
        plot_handles["t_hist"].pop(0)
        plot_handles["theta_hist"].pop(0)
        plot_handles["theta_ref_hist"].pop(0)
        plot_handles["phi_hist"].pop(0)
        plot_handles["phi_ref_hist"].pop(0)

    plot_handles["line_theta"].set_data(plot_handles["t_hist"], plot_handles["theta_hist"])
    plot_handles["line_theta_ref"].set_data(plot_handles["t_hist"], plot_handles["theta_ref_hist"])
    plot_handles["line_phi"].set_data(plot_handles["t_hist"], plot_handles["phi_hist"])
    plot_handles["line_phi_ref"].set_data(plot_handles["t_hist"], plot_handles["phi_ref_hist"])

    for a in plot_handles["ax"]:
        a.relim()
        a.autoscale_view()

    plt.pause(0.001)
    return True


# =========================================================
# Main
# =========================================================
def main():
    sigma_rad = np.deg2rad(SIGMA_DEG)

    phi_ref_rad = math.radians(PHI_REF_DEG)
    theta_ref_rad = math.radians(THETA_REF_DEG)

    mapper = CurrentTensionMapper(
        r_pulley_m=R_PULLEY_M,
        tension_to_current_gain_mA_per_N=A_MA_PER_N,
        tension_to_current_offset_mA=B_MA,
        tighten_sign=TIGHTEN_SIGN,
        i_max_mA=I_MAX_MA,
    )

    imu = None
    dxl = None
    plot_handles = None

    try:
        # -----------------------------
        # IMU init
        # -----------------------------
        imu = IMUBendReader(port=IMU_PORT, baudrate=115200, timeout=0.02)

        print("=" * 72)
        print("CLOSED-LOOP BEND -> TENSION -> CURRENT CONTROL (PID + live plot)")
        print("=" * 72)
        print("Keep the arm straight and unloaded for reference capture.")
        input("Press Enter to capture straight reference... ")

        imu.capture_straight_reference(n=60, dt=0.01)

        # -----------------------------
        # Dynamixel init
        # -----------------------------
        motor_models = [BaseModel(motor_id=mid) for mid in MOTOR_IDS]
        dxl = DynamixelController(
            port_name=DXL_PORT,
            motor_list=motor_models,
            protocol=DXL_PROTOCOL,
            baudrate=DXL_BAUDRATE,
            reverse_direction=REVERSE_DIRECTION,
        )

        print("Connecting to Dynamixels...")
        dxl.activate_controller()
        dxl.torque_off()
        dxl.set_operating_mode_all("current_control")
        dxl.torque_on()
        safe_zero_currents(dxl, len(MOTOR_IDS))
        time.sleep(0.3)

        if ENABLE_LIVE_PLOT:
            plot_handles = init_live_plot()

        print()
        print(f"Target: phi={PHI_REF_DEG:.1f} deg, theta={THETA_REF_DEG:.1f} deg")
        print("Close the plot window or press Ctrl+C to stop.")
        print()

        bend_int = np.zeros(2, dtype=float)
        prev_e_u = np.zeros(2, dtype=float)
        e_dot_filt = np.zeros(2, dtype=float)

        last_print_t = 0.0
        last_plot_t = 0.0

        while True:
            t0 = time.time()

            # -----------------------------
            # Read IMU
            # -----------------------------
            state = imu.get_state(
                phi_offset_deg=PHI_OFFSET_DEG,
                phi_valid_thresh_deg=PHI_VALID_THRESH_DEG,
            )

            phi_meas_rad = float(state["phi_rad"])
            theta_meas_rad = float(state["theta_rad"])
            phi_valid = bool(state["phi_valid"])

            theta_meas_deg = float(state["theta_deg"])
            phi_meas_deg = float(state["phi_deg"])

            # -----------------------------
            # Safety
            # -----------------------------
            if theta_meas_deg > THETA_HARD_LIMIT_DEG:
                print("WARNING: theta hard limit exceeded. Sending zero current.")
                safe_zero_currents(dxl, len(MOTOR_IDS))
                bend_int[:] = 0.0
                prev_e_u[:] = 0.0
                e_dot_filt[:] = 0.0
                time.sleep(0.1)
                continue

            # -----------------------------
            # Outer loop: bend error -> bend command
            # -----------------------------
            u_cmd_p, e_u, u_ref, u_meas = bend_command_from_bend_error(
                phi_ref_rad=phi_ref_rad,
                theta_ref_rad=theta_ref_rad,
                phi_meas_rad=phi_meas_rad,
                theta_meas_rad=theta_meas_rad,
                k_bend=K_BEND,
                phi_valid=phi_valid,
                theta_valid_thresh_rad=math.radians(PHI_VALID_THRESH_DEG),
            )

            # Integral
            bend_int += e_u * DT
            bend_int = clamp_vec(bend_int, -INT_LIM, INT_LIM)

            # Derivative with filtering
            e_dot_raw = (e_u - prev_e_u) / DT
            e_dot_filt = (1.0 - D_FILT_ALPHA) * e_dot_filt + D_FILT_ALPHA * e_dot_raw
            prev_e_u = e_u.copy()

            # PID bend command
            u_cmd = u_cmd_p + KI_BEND * bend_int + KD_BEND * e_dot_filt

            # -----------------------------
            # Bend command -> tendon tensions
            # -----------------------------
            tensions_N = solve_tensions_single(
                u_bend_cmd=u_cmd,
                sigma_rad=sigma_rad,
                t_bias=T_BIAS_N,
                t_min=T_MIN_N,
                t_max=T_MAX_N,
            )

            # -----------------------------
            # Tensions -> motor currents
            # -----------------------------
            current_cmd_signed_mA = mapper.tensions_to_signed_currents_mA(tensions_N)
            dxl.set_goal_current_mA(current_cmd_signed_mA.tolist())

            # -----------------------------
            # Read motors for monitoring
            # -----------------------------
            pos_deg, vel_deg_s, cur_mA, pwm_pct = dxl.read_info_with_unit(
                pwm_unit="percent",
                angle_unit="deg",
                current_unit="mA",
                fast_read=True,
            )

            t_est = np.array(
                [mapper.current_to_tension_N(cur_mA[i], i) for i in range(len(MOTOR_IDS))],
                dtype=float,
            )

            # -----------------------------
            # Print
            # -----------------------------
            now = time.time()
            if now - last_print_t >= PRINT_DT:
                phi_text = f"{wrap_to_360(phi_meas_deg):7.2f}" if phi_valid else "invalid"
                print(
                    f"theta={theta_meas_deg:6.2f} deg | "
                    f"phi={phi_text} | "
                    f"e_u=[{e_u[0]:+.3f}, {e_u[1]:+.3f}] | "
                    f"int=[{bend_int[0]:+.3f}, {bend_int[1]:+.3f}] | "
                    f"d=[{e_dot_filt[0]:+.3f}, {e_dot_filt[1]:+.3f}] | "
                    f"T_ref={np.round(tensions_N, 3)} N | "
                    f"T_est={np.round(t_est, 3)} N | "
                    f"I_cmd={np.round(current_cmd_signed_mA, 1)} mA | "
                    f"I_meas={np.round(cur_mA, 1)} mA"
                )
                last_print_t = now

            # -----------------------------
            # Live plot
            # -----------------------------
            if ENABLE_LIVE_PLOT and (now - last_plot_t >= PLOT_DT):
                still_open = update_live_plot(
                    plot_handles=plot_handles,
                    theta_deg=theta_meas_deg,
                    phi_deg=phi_meas_deg,
                    phi_valid=phi_valid,
                    theta_ref_deg=THETA_REF_DEG,
                    phi_ref_deg=PHI_REF_DEG,
                )
                last_plot_t = now

                if not still_open:
                    print("Plot window closed. Stopping controller.")
                    break

            # -----------------------------
            # Timing
            # -----------------------------
            elapsed = time.time() - t0
            time.sleep(max(0.0, DT - elapsed))

    except PortCommError as e:
        print("\nDynamixel communication failed.")
        print(str(e))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        if dxl is not None:
            try:
                safe_zero_currents(dxl, len(MOTOR_IDS))
                time.sleep(0.2)
                dxl.torque_off()
            except Exception:
                pass

        if imu is not None:
            try:
                imu.close()
            except Exception:
                pass

        try:
            plt.ioff()
            plt.close("all")
        except Exception:
            pass

        print("Exited safely.")


if __name__ == "__main__":
    main()