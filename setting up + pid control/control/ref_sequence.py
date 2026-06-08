import math
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from b1_phi_theta import IMUBendReader
from b3_tendon_mapping import bend_command_from_bend_error, solve_tensions_single
from b2_current_actuator import CurrentTensionMapper
from dynamixel_controller import DynamixelController, BaseModel, PortCommError


IMU_PORT = "COM5"
DXL_PORT = "COM4"

MOTOR_IDS = [1, 2, 3]
DXL_BAUDRATE = 57600
DXL_PROTOCOL = 2.0
REVERSE_DIRECTION = False

SIGMA_DEG = np.array([180.0, -45.0, 70.0], dtype=float)

# ---------------------------------------------------------
# Reference sequence
# ---------------------------------------------------------
REF_SEQUENCE = [
    {"name": "Hold straight",   "phi_deg": float(SIGMA_DEG[0]), "theta_deg": 0.0,  "duration": 3.0},
    {"name": "Step to M1",      "phi_deg": float(SIGMA_DEG[0]), "theta_deg": 10.0, "duration": 7.0},
    {"name": "Step to M2",      "phi_deg": float(SIGMA_DEG[1]), "theta_deg": 15.0,  "duration": 7.0},
    {"name": "Between M2 and M3", "phi_deg": 12.5, "theta_deg": 8.0, "duration": 6.0},
    {"name": "Return straight", "phi_deg": float(SIGMA_DEG[1]), "theta_deg": 0.0,  "duration": 6.0},
]

REPEAT_SEQUENCE = False
RESET_PID_ON_PHASE_CHANGE = False

# Current / tension mapping
R_PULLEY_M = 0.009
A_MA_PER_N = 64.7
B_MA = 0.0
TIGHTEN_SIGN = (-1.0, -1.0, -1.0)

# Closed-loop bend gains
K_BEND = np.array([3.3, 3.3], dtype=float)
KI_BEND = np.array([0.25, 0.25], dtype=float)
KD_BEND = np.array([0.03, 0.03], dtype=float)

# Integral / derivative handling
INT_LIM = np.array([3.0, 3.0], dtype=float)
D_FILT_ALPHA = 0.15

# Tension limits
T_BIAS_BENT_N = 0.40
T_BIAS_STRAIGHT_N = 0.25
T_MIN_N = 0.0
T_MAX_N = 3.0

PRETENSION_BEFORE_CAPTURE = True
PRETENSION_RAMP_STEPS = 50
PRETENSION_RAMP_DT = 0.03
PRETENSION_SETTLE_SEC = 1.5

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

# Save plot
SAVE_PLOT = True
SAVE_DIR = "results"
SAVE_DPI = 180


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


def get_phase_bias(theta_ref_deg):
    if abs(theta_ref_deg) < 0.5:
        return T_BIAS_STRAIGHT_N
    return T_BIAS_BENT_N


def get_active_reference(sequence, t_elapsed, repeat=False):
    durations = [phase["duration"] for phase in sequence]
    total_time = sum(durations)

    if total_time <= 0.0:
        return None, None, True

    if repeat:
        t_eval = t_elapsed % total_time
        finished = False
    else:
        if t_elapsed >= total_time:
            return None, None, True
        t_eval = t_elapsed
        finished = False

    t_cum = 0.0
    for i, phase in enumerate(sequence):
        t_cum += phase["duration"]
        if t_eval < t_cum:
            return i, phase, finished

    return None, None, True


def command_equal_pretension(dxl, mapper, n_motors, tension_per_tendon_N):
    tensions = np.full(n_motors, tension_per_tendon_N, dtype=float)
    current_cmd_signed_mA = mapper.tensions_to_signed_currents_mA(tensions)
    dxl.set_goal_current_mA(current_cmd_signed_mA.tolist())
    return tensions, current_cmd_signed_mA


def ramp_equal_pretension(dxl, mapper, n_motors, target_tension_per_tendon_N, steps, dt):
    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        t_now = alpha * target_tension_per_tendon_N
        command_equal_pretension(
            dxl=dxl,
            mapper=mapper,
            n_motors=n_motors,
            tension_per_tendon_N=t_now,
        )
        time.sleep(dt)


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

    fig.suptitle("Closed-loop bend tracking with reference sequence")
    fig.tight_layout()

    return {
        "fig": fig,
        "ax": ax,
        "line_theta": line_theta,
        "line_theta_ref": line_theta_ref,
        "line_phi": line_phi,
        "line_phi_ref": line_phi_ref,

        # rolling window for live display
        "t_hist": [],
        "theta_hist": [],
        "theta_ref_hist": [],
        "phi_hist": [],
        "phi_ref_hist": [],

        # full history for saving
        "full_t_hist": [],
        "full_theta_hist": [],
        "full_theta_ref_hist": [],
        "full_phi_hist": [],
        "full_phi_ref_hist": [],

        # keep last valid phi so the curve does not disappear
        "last_valid_phi_deg": np.nan,

        "plot_t0": time.time(),
    }


def update_live_plot(plot_handles, theta_deg, phi_deg, phi_valid, theta_ref_deg, phi_ref_deg):
    if plot_handles is None:
        return False

    fig = plot_handles["fig"]
    if not plt.fignum_exists(fig.number):
        return False

    t_now = time.time() - plot_handles["plot_t0"]

    phi_raw = wrap_to_360(phi_deg)
    if phi_valid:
        phi_plot = phi_raw
        plot_handles["last_valid_phi_deg"] = phi_raw
    else:
        phi_plot = plot_handles["last_valid_phi_deg"]

    phi_ref_plot = wrap_to_360(phi_ref_deg)

    # full history for saving
    plot_handles["full_t_hist"].append(t_now)
    plot_handles["full_theta_hist"].append(theta_deg)
    plot_handles["full_theta_ref_hist"].append(theta_ref_deg)
    plot_handles["full_phi_hist"].append(phi_plot)
    plot_handles["full_phi_ref_hist"].append(phi_ref_plot)

    # rolling history for live plot
    plot_handles["t_hist"].append(t_now)
    plot_handles["theta_hist"].append(theta_deg)
    plot_handles["theta_ref_hist"].append(theta_ref_deg)
    plot_handles["phi_hist"].append(phi_plot)
    plot_handles["phi_ref_hist"].append(phi_ref_plot)

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


def save_performance_plot(plot_handles, save_dir=SAVE_DIR, dpi=SAVE_DPI):
    if plot_handles is None:
        return None

    t = plot_handles["full_t_hist"]
    if len(t) == 0:
        return None

    theta = plot_handles["full_theta_hist"]
    theta_ref = plot_handles["full_theta_ref_hist"]
    phi = plot_handles["full_phi_hist"]
    phi_ref = plot_handles["full_phi_ref_hist"]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = save_path / f"bend_tracking_{timestamp}.png"

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax[0].plot(t, theta, label="theta")
    ax[0].plot(t, theta_ref, "--", label="theta_ref")
    ax[0].set_ylabel("Theta [deg]")
    ax[0].grid(True)
    ax[0].legend(loc="upper right")

    ax[1].plot(t, phi, label="phi")
    ax[1].plot(t, phi_ref, "--", label="phi_ref")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Phi [deg]")
    ax[1].grid(True)
    ax[1].legend(loc="upper right")

    fig.suptitle("Closed-loop bend tracking performance")
    fig.tight_layout()
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return file_path


# =========================================================
# Main
# =========================================================
def main():
    sigma_rad = np.deg2rad(SIGMA_DEG)

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

        # -----------------------------
        # Dynamixel init FIRST
        # -----------------------------
        motor_models = [BaseModel(motor_id=mid) for mid in MOTOR_IDS]
        dxl = DynamixelController(
            port_name=DXL_PORT,
            motor_list=motor_models,
            protocol=DXL_PROTOCOL,
            baudrate=DXL_BAUDRATE,
            reverse_direction=REVERSE_DIRECTION,
        )

        print("=" * 72)
        print("CLOSED-LOOP BEND -> TENSION -> CURRENT CONTROL (REFERENCE SEQUENCE)")
        print("=" * 72)
        print("Connecting to Dynamixels...")
        dxl.activate_controller()
        dxl.torque_off()
        dxl.set_operating_mode_all("current_control")
        dxl.torque_on()
        safe_zero_currents(dxl, len(MOTOR_IDS))
        time.sleep(0.3)

        # -----------------------------
        # Pretension BEFORE straight capture
        # -----------------------------
        pretension_for_capture = T_BIAS_BENT_N

        if PRETENSION_BEFORE_CAPTURE:
            print()
            print(f"Ramping equal pretension to {pretension_for_capture:.3f} N per tendon...")
            ramp_equal_pretension(
                dxl=dxl,
                mapper=mapper,
                n_motors=len(MOTOR_IDS),
                target_tension_per_tendon_N=pretension_for_capture,
                steps=PRETENSION_RAMP_STEPS,
                dt=PRETENSION_RAMP_DT,
            )
            print(f"Holding pretension for {PRETENSION_SETTLE_SEC:.1f} s to settle...")
            time.sleep(PRETENSION_SETTLE_SEC)
            print()
            print("Now keep the arm physically straight UNDER pretension.")
            input("Press Enter to capture straight reference under preload... ")
        else:
            print()
            print("Keep the arm straight and unloaded for reference capture.")
            input("Press Enter to capture straight reference... ")

        imu.capture_straight_reference(n=60, dt=0.01)
        print("Straight reference captured.")

        if ENABLE_LIVE_PLOT:
            plot_handles = init_live_plot()

        print("\nReference sequence:")
        for i, phase in enumerate(REF_SEQUENCE):
            print(
                f"  {i+1}. {phase['name']}: "
                f"phi={phase['phi_deg']:.1f} deg, "
                f"theta={phase['theta_deg']:.1f} deg, "
                f"duration={phase['duration']:.1f} s"
            )
        print("\nClose the plot window or press Ctrl+C to stop.\n")

        # controller states
        bend_int = np.zeros(2, dtype=float)
        prev_e_u = np.zeros(2, dtype=float)
        e_dot_filt = np.zeros(2, dtype=float)

        last_print_t = 0.0
        last_plot_t = 0.0
        seq_t0 = time.time()
        prev_phase_idx = None

        while True:
            t_loop_start = time.time()
            t_seq = t_loop_start - seq_t0

            # -----------------------------
            # Reference scheduler
            # -----------------------------
            phase_idx, phase, finished = get_active_reference(
                REF_SEQUENCE, t_seq, repeat=REPEAT_SEQUENCE
            )

            if finished:
                print("Reference sequence finished.")
                break

            if phase_idx != prev_phase_idx:
                print(
                    f"\n---- Active reference: {phase['name']} | "
                    f"phi_ref={phase['phi_deg']:.1f} deg, "
                    f"theta_ref={phase['theta_deg']:.1f} deg ----"
                )
                if RESET_PID_ON_PHASE_CHANGE:
                    bend_int[:] = 0.0
                    prev_e_u[:] = 0.0
                    e_dot_filt[:] = 0.0
                prev_phase_idx = phase_idx

            phi_ref_deg = float(phase["phi_deg"])
            theta_ref_deg = float(phase["theta_deg"])
            phi_ref_rad = math.radians(phi_ref_deg)
            theta_ref_rad = math.radians(theta_ref_deg)

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
            t_bias_now = get_phase_bias(theta_ref_deg)

            tensions_N = solve_tensions_single(
                u_bend_cmd=u_cmd,
                sigma_rad=sigma_rad,
                t_bias=t_bias_now,
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
                    f"[{phase['name']}] "
                    f"theta={theta_meas_deg:6.2f} deg | "
                    f"phi={phi_text} | "
                    f"theta_ref={theta_ref_deg:5.1f} | "
                    f"phi_ref={wrap_to_360(phi_ref_deg):6.1f} | "
                    f"bias={t_bias_now:4.2f} N | "
                    f"e_u=[{e_u[0]:+.3f}, {e_u[1]:+.3f}] | "
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
                    theta_ref_deg=theta_ref_deg,
                    phi_ref_deg=phi_ref_deg,
                )
                last_plot_t = now

                if not still_open:
                    print("Plot window closed. Stopping controller.")
                    break

            # -----------------------------
            # Timing
            # -----------------------------
            elapsed = time.time() - t_loop_start
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

        if SAVE_PLOT and plot_handles is not None:
            try:
                saved_file = save_performance_plot(plot_handles)
                if saved_file is not None:
                    print(f"Saved performance plot to: {saved_file}")
                else:
                    print("No plot data available to save.")
            except Exception as e:
                print(f"Could not save performance plot: {e}")

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