import math
import time
import numpy as np

from b1_phi_theta import IMUBendReader
from b3_tendon_mapping import bend_command_from_bend_error, solve_tensions_single
from b2_current_actuator import CurrentTensionMapper
from dynamixel_controller import DynamixelController, BaseModel, PortCommError


# =========================================================
# USER SETTINGS
# =========================================================
IMU_PORT = "COM5"
DXL_PORT = "COM4"

MOTOR_IDS = [1, 2, 3]
DXL_BAUDRATE = 57600
DXL_PROTOCOL = 2.0
REVERSE_DIRECTION = False

# Desired bend target
PHI_REF_DEG = 180.0
THETA_REF_DEG = 15.0

SIGMA_DEG = np.array([180.0, -45.0, 70.0], dtype=float)

# Current / tension mapping
R_PULLEY_M = 0.009
A_MA_PER_N = 64.7          
B_MA = 0.0               
TIGHTEN_SIGN = (-1.0, -1.0, -1.0)

# Closed-loop bend gain
K_BEND = np.array([3.0, 3.0], dtype=float)
KI_BEND = np.array([0.2, 0.2], dtype=float)
INT_LIM = np.array([0.2, 0.2], dtype=float)

# Tension limits
T_BIAS_N = 0.0
T_MIN_N = 0.0
T_MAX_N = 3.0

# Current safety
I_MAX_MA = 120.0

# Angle safety
THETA_HARD_LIMIT_DEG = 35.0
PHI_VALID_THRESH_DEG = 3.0
PHI_OFFSET_DEG = 0.0

# Loop timing
DT = 0.05
PRINT_DT = 0.10

class IdentifiedCurrentTensionMapper:
    """
    Simple identified mapping:
        I_mA = A_MA_PER_N * T_N + B_MA

    and inverse:
        T_N = (I_mA - B_MA) / A_MA_PER_N
    """

    def __init__(
        self,
        a_mA_per_N,
        b_mA,
        r_pulley_m=0.009,
        tighten_sign=(-1.0, -1.0, -1.0),
        i_max_mA=120.0,
    ):
        self.a_mA_per_N = float(a_mA_per_N)
        self.b_mA = float(b_mA)
        self.r_pulley_m = float(r_pulley_m)
        self.tighten_sign = np.asarray(tighten_sign, dtype=float)
        self.i_max_mA = float(i_max_mA)

    def tension_to_current_mA(self, tension_N: float) -> float:
        tension_N = max(0.0, float(tension_N))
        if tension_N <= 1e-12:
            return 0.0
        i_mA = self.a_mA_per_N * tension_N + self.b_mA
        return max(0.0, i_mA)

    def tensions_to_signed_currents_mA(self, tensions_N):
        tensions_N = np.asarray(tensions_N, dtype=float)
        mags = np.array([self.tension_to_current_mA(t) for t in tensions_N], dtype=float)
        mags = np.clip(mags, 0.0, self.i_max_mA)
        return self.tighten_sign[:len(mags)] * mags

    def measured_current_to_tension_N(self, current_mA, motor_index):
        sign = self.tighten_sign[motor_index]
        i_tight_mA = max(0.0, sign * float(current_mA))
        if i_tight_mA <= self.b_mA:
            return 0.0
        return max(0.0, (i_tight_mA - self.b_mA) / self.a_mA_per_N)


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


# =========================================================
# Main
# =========================================================
def main():
    sigma_rad = np.deg2rad(SIGMA_DEG)

    phi_ref_rad = math.radians(PHI_REF_DEG)
    theta_ref_rad = math.radians(THETA_REF_DEG)

    mapper = IdentifiedCurrentTensionMapper(
        a_mA_per_N=A_MA_PER_N,
        b_mA=B_MA,
        r_pulley_m=R_PULLEY_M,
        tighten_sign=TIGHTEN_SIGN,
        i_max_mA=I_MAX_MA,
    )

    imu = None
    dxl = None

    try:
        # -----------------------------
        # IMU init
        # -----------------------------
        imu = IMUBendReader(port=IMU_PORT, baudrate=115200, timeout=0.02)

        print("=" * 72)
        print("CLOSED-LOOP BEND -> TENSION -> CURRENT CONTROL")
        print("=" * 72)
        print("This starts as a P controller.")
        print("Keep the arm straight and free.")
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

        print()
        print(f"Target: phi={PHI_REF_DEG:.1f} deg, theta={THETA_REF_DEG:.1f} deg")
        print("Ctrl+C to stop.")
        print()

        bend_int = np.zeros(2, dtype=float)
        last_print_t = 0.0

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

            # Safety
            if theta_meas_deg > THETA_HARD_LIMIT_DEG:
                print("WARNING: theta hard limit exceeded. Sending zero current.")
                safe_zero_currents(dxl, len(MOTOR_IDS))
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

            bend_int += e_u * DT
            bend_int = clamp_vec(bend_int, -INT_LIM, INT_LIM)

            u_cmd = u_cmd_p + KI_BEND * bend_int

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
                [mapper.measured_current_to_tension_N(cur_mA[i], i) for i in range(len(MOTOR_IDS))],
                dtype=float,
            )

            # -----------------------------
            # Print
            # -----------------------------
            now = time.time()
            if now - last_print_t >= PRINT_DT:
                phi_text = f"{phi_meas_deg:+7.2f}" if phi_valid else "   invalid"
                print(
                    f"theta={theta_meas_deg:6.2f} deg | "
                    f"phi={phi_text} | "
                    f"e_u=[{e_u[0]:+.3f}, {e_u[1]:+.3f}] | "
                    f"T_ref={np.round(tensions_N, 3)} N | "
                    f"T_est={np.round(t_est, 3)} N | "
                    f"I_cmd={np.round(current_cmd_signed_mA, 1)} mA | "
                    f"I_meas={np.round(cur_mA, 1)} mA"
                )
                last_print_t = now

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

        print("Exited safely.")


if __name__ == "__main__":
    main()