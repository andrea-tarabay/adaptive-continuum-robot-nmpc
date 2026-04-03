import math
import time
import serial
from typing import Optional, List, Dict

import numpy as np


# =========================================================
# Basic helpers
# =========================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# =========================================================
# IMU reader
# =========================================================
class IMUBendReader:
    """
    Reads RAW quaternion data from Arduino/BNO055.

    Expected Arduino line format:
        Q,w,x,y,z

    Processing pipeline:
        raw BNO quaternion
        -> fixed IMU-to-arm mounting correction
        -> software straight reference capture
        -> relative arm quaternion
        -> PCC extraction 
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.02,
    ):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(2.0)  # allow Arduino reset

        
        # current mounting estimate:
        #   x_arm =  x_imu
        #   y_arm =  z_imu
        #   z_arm = -y_imu
        
        self.R_AI = np.array([
            [1.0,  0.0,  0.0],   # x_arm =  x_imu
            [0.0,  0.0,  1.0],   # y_arm =  z_imu
            [0.0, -1.0,  0.0],   # z_arm = -y_imu
        ], dtype=float)

        # Arm -> IMU rotation
        self.R_IA = self.R_AI.T

        # Straight reference (set later)
        self.R_WA_ref: Optional[np.ndarray] = None

    # -------------------------------------------------
    # Quaternion helpers
    # Convention in code: q = [w, x, y, z]
    # -------------------------------------------------
    @staticmethod
    def quat_normalize(q: List[float]) -> List[float]:
        n = math.sqrt(sum(v * v for v in q))
        if n < 1e-12:
            return [1.0, 0.0, 0.0, 0.0]
        return [v / n for v in q]

    @staticmethod
    def quat_conj(q: List[float]) -> List[float]:
        w, x, y, z = q
        return [w, -x, -y, -z]

    @staticmethod
    def quat_mul(q1: List[float], q2: List[float]) -> List[float]:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]

    @classmethod
    def quat_relative(cls, q_ref: List[float], q_now: List[float]) -> List[float]:
        """
        q_rel = conj(q_ref) * q_now
        """
        return cls.quat_normalize(cls.quat_mul(cls.quat_conj(q_ref), q_now))

    @staticmethod
    def quat_dot(q1: List[float], q2: List[float]) -> float:
        return sum(a * b for a, b in zip(q1, q2))

    @staticmethod
    def quat_to_rotmat(q: List[float]) -> np.ndarray:
        """
        Quaternion [w, x, y, z] -> rotation matrix R.
        R maps body-frame vectors into world-frame vectors.
        """
        w, x, y, z = IMUBendReader.quat_normalize(q)

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        R = np.array([
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ], dtype=float)
        return R

    @staticmethod
    def rotmat_to_quat(R: np.ndarray) -> List[float]:
        """
        Rotation matrix -> quaternion [w, x, y, z]
        """
        tr = float(np.trace(R))

        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return IMUBendReader.quat_normalize([w, x, y, z])

    # -------------------------------------------------
    # Serial parsing
    # -------------------------------------------------
    def _parse_q_line(self, line: str) -> Optional[List[float]]:
        if not line:
            return None

        if line.startswith("READY") or line.startswith("INFO") or line.startswith("ERR"):
            return None

        parts = line.split(",")
        if len(parts) != 5:
            return None

        if parts[0] != "Q":
            return None

        try:
            w = float(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
        except ValueError:
            return None

        return self.quat_normalize([w, x, y, z])

    def read_raw_quaternion(self) -> List[float]:
        """
        Return the newest valid Q sample available.
        """
        last_q = None

        while self.ser.in_waiting > 0:
            line = self.ser.readline().decode(errors="ignore").strip()
            q = self._parse_q_line(line)
            if q is not None:
                last_q = q

        if last_q is not None:
            return last_q

        while True:
            line = self.ser.readline().decode(errors="ignore").strip()
            q = self._parse_q_line(line)
            if q is not None:
                return q

    def get_state(
        self,
        phi_offset_deg: float = 0.0,
        phi_valid_thresh_deg: float = 3.0,
    ) -> Dict[str, object]:
        state = self.get_relative_arm_quaternion()
        q_rel = state["q_rel_wxyz"]

        pcc = self.pcc_from_quaternion(
            q_rel,
            phi_offset_deg=phi_offset_deg,
            phi_valid_thresh_deg=phi_valid_thresh_deg,
        )

        return {
            "q_raw_wxyz": [float(v) for v in state["q_raw_wxyz"]],
            "q_rel_wxyz": [float(v) for v in state["q_rel_wxyz"]],
            "phi_rad": float(pcc["phi_rad"]),
            "phi_deg": float(pcc["phi_deg"]),
            "theta_rad": float(pcc["theta_rad"]),
            "theta_deg": float(pcc["theta_deg"]),
            "phi_valid": bool(pcc["phi_valid"]),
        }
    # -------------------------------------------------
    # Averaging
    # -------------------------------------------------
    def average_quaternion_samples(self, n: int = 40, dt: float = 0.01) -> List[float]:
        """
        Average raw quaternion samples with sign alignment.
        """
        qs = []
        q_sign_ref = None

        for _ in range(n):
            q = self.read_raw_quaternion()

            if q_sign_ref is None:
                q_sign_ref = q
            else:
                if self.quat_dot(q, q_sign_ref) < 0.0:
                    q = [-v for v in q]

            qs.append(q)
            time.sleep(dt)

        q_mean = [sum(q[i] for q in qs) / len(qs) for i in range(4)]
        return self.quat_normalize(q_mean)

    # -------------------------------------------------
    # Frame correction
    # -------------------------------------------------
    def raw_quat_to_arm_rotmat(self, q_raw_wxyz: List[float]) -> np.ndarray:
        """
        Convert raw IMU quaternion to corrected ARM orientation matrix.

        q_raw defines IMU orientation in world.
        R_WI: IMU frame -> world frame
        R_IA: ARM frame -> IMU frame

        Then:
            R_WA = R_WI @ R_IA
        """
        R_WI = self.quat_to_rotmat(q_raw_wxyz)
        R_WA = R_WI @ self.R_IA
        return R_WA

    def raw_quat_to_arm_quat(self, q_raw_wxyz: List[float]) -> List[float]:
        R_WA = self.raw_quat_to_arm_rotmat(q_raw_wxyz)
        return self.rotmat_to_quat(R_WA)

    def capture_straight_reference(self, n: int = 60, dt: float = 0.01) -> None:
        """
        Capture and store the straight-pose reference in ARM frame.
        """
        print("Hold the arm in the pose you define as STRAIGHT...")
        time.sleep(1.0)

        q_raw_mean = self.average_quaternion_samples(n=n, dt=dt)
        self.R_WA_ref = self.raw_quat_to_arm_rotmat(q_raw_mean)

        print("Straight reference captured.")
        print(f"q_raw_ref = {[round(v, 6) for v in q_raw_mean]}")

    def get_relative_arm_quaternion(self) -> Dict[str, List[float]]:
        """
        Returns:
            raw quaternion
            corrected arm quaternion in world
            relative arm quaternion wrt straight reference
        """
        if self.R_WA_ref is None:
            raise RuntimeError("Straight reference not captured yet.")

        q_raw = self.read_raw_quaternion()
        R_WA_now = self.raw_quat_to_arm_rotmat(q_raw)

        # Relative arm rotation: reference^T * current
        R_rel = self.R_WA_ref.T @ R_WA_now
        q_rel = self.rotmat_to_quat(R_rel)
        q_arm_now = self.rotmat_to_quat(R_WA_now)

        return {
            "q_raw_wxyz": [float(v) for v in q_raw],
            "q_arm_now_wxyz": [float(v) for v in q_arm_now],
            "q_rel_wxyz": [float(v) for v in q_rel],
        }

    # -------------------------------------------------
    # PCC extraction 
    # -------------------------------------------------
    @staticmethod
    def pcc_from_quaternion(
        q_wxyz: List[float],
        phi_offset_deg: float = 0.0,
        phi_valid_thresh_deg: float = 3.0,
    ) -> Dict[str, float]:
        """
       
        Paper notation:
            Q = [xq, yq, zq, wq]^T
            phi   = atan2(xq*zq - wq*yq, yq*zq + wq*xq)
            theta = acos(2*wq^2 - 1 + 2*zq^2)

        Here input format is [w, x, y, z].

        Returns phi/theta in rad/deg and a phi_valid flag.
        """
        wq, xq, yq, zq = IMUBendReader.quat_normalize(q_wxyz)

        phi = math.atan2(
            xq * zq - wq * yq,
            yq * zq + wq * xq,
        )
        phi = wrap_to_pi(phi + math.radians(phi_offset_deg))

        theta = math.acos(clamp(2.0 * wq * wq - 1.0 + 2.0 * zq * zq, -1.0, 1.0))

        phi_valid = theta > math.radians(phi_valid_thresh_deg)

        return {
            "phi_rad": phi,
            "phi_deg": math.degrees(phi),
            "theta_rad": theta,
            "theta_deg": math.degrees(theta),
            "phi_valid": phi_valid,
        }

    def close(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()


# =========================================================
# Main live test
# =========================================================
if __name__ == "__main__":
    imu = IMUBendReader(port="COM4", baudrate=115200, timeout=0.02)

    try:
        print("=" * 70)
        print("IMU -> ARM corrected PCC test")
        print("Arduino must be sending RAW lines: Q,w,x,y,z")
        
        print("=" * 70)
        print()
        print("Step 1: keep the segment in the pose you define as STRAIGHT.")
        input("Press Enter to capture the straight reference...")

        imu.capture_straight_reference(n=60, dt=0.01)

        print()
        print("Streaming live values.")
        print("Ctrl+C to stop.")
        print()

        last_print_t = 0.0
        print_dt = 0.10

        while True:
            state = imu.get_relative_arm_quaternion()
            q_rel = state["q_rel_wxyz"]

            pcc = imu.pcc_from_quaternion(
                q_rel,
                phi_offset_deg=0.0,
                phi_valid_thresh_deg=3.0,
            )

            now_t = time.time()
            if now_t - last_print_t >= print_dt:
                print(
                    f"theta={pcc['theta_deg']:6.2f} deg | "
                    f"phi={pcc['phi_deg']:+7.2f} deg | "
                    f"valid={pcc['phi_valid']}"
                )
                last_print_t = now_t

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        imu.close()
        print("Exited safely.")