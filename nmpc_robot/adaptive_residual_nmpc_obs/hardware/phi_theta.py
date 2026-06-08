import math
import time
import serial
import numpy as np


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class IMUBendReader:
    """
    Read raw quaternion lines from Arduino:
        Q,w,x,y,z

    Then:
    - apply IMU-to-arm frame correction
    - capture a straight reference
    - compute relative arm rotation
    - extract phi and theta
    """

    def __init__(self, port, baudrate=115200, timeout=0.02):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        time.sleep(2.0)  # Arduino reset time

        # IMU frame -> arm frame
        self.R_AI = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ], dtype=float)

        # arm frame -> IMU frame
        self.R_IA = self.R_AI.T

        self.R_WA_ref = None

    # =====================================================
    # Quaternion helpers
    # =====================================================
    def quat_normalize(self, q):
        q = np.asarray(q, dtype=float)
        n = np.linalg.norm(q)
        if n < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return q / n

    def quat_dot(self, q1, q2):
        return float(np.dot(q1, q2))

    def quat_to_rotmat(self, q):
        w, x, y, z = self.quat_normalize(q)

        R = np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
        ], dtype=float)

        return R

    def rotmat_to_quat(self, R):
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

        return self.quat_normalize([w, x, y, z])

    # =====================================================
    # Serial read
    # =====================================================
    def _parse_q_line(self, line):
        if not line:
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

    def read_raw_quaternion(self):
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

    # =====================================================
    # Frame conversion
    # =====================================================
    def raw_quat_to_arm_rotmat(self, q_raw):
        R_WI = self.quat_to_rotmat(q_raw)
        R_WA = R_WI @ self.R_IA
        return R_WA

    def average_quaternion_samples(self, n=40, dt=0.01):
        qs = []
        q_ref = None

        for _ in range(n):
            q = self.read_raw_quaternion()

            if q_ref is None:
                q_ref = q
            else:
                if self.quat_dot(q, q_ref) < 0.0:
                    q = -q

            qs.append(q)
            time.sleep(dt)

        q_mean = np.mean(np.array(qs), axis=0)
        return self.quat_normalize(q_mean)

    def capture_straight_reference(self, n=60, dt=0.01):
        print("Hold the arm in the pose you define as STRAIGHT...")
        time.sleep(1.0)

        q_raw_mean = self.average_quaternion_samples(n=n, dt=dt)
        self.R_WA_ref = self.raw_quat_to_arm_rotmat(q_raw_mean)

        print("Straight reference captured.")
        print("q_raw_ref =", [round(v, 6) for v in q_raw_mean])

    def get_relative_arm_quaternion(self):
        if self.R_WA_ref is None:
            raise RuntimeError("Straight reference not captured yet.")

        q_raw = self.read_raw_quaternion()
        R_WA_now = self.raw_quat_to_arm_rotmat(q_raw)

        # relative rotation = reference^T * current
        R_rel = self.R_WA_ref.T @ R_WA_now
        q_rel = self.rotmat_to_quat(R_rel)

        return {
            "q_raw_wxyz": [float(v) for v in q_raw],
            "q_rel_wxyz": [float(v) for v in q_rel],
        }

    # =====================================================
    # PCC angles
    # =====================================================
    def pcc_from_quaternion(self, q_wxyz, phi_offset_deg=0.0, phi_valid_thresh_deg=3.0):
        wq, xq, yq, zq = self.quat_normalize(q_wxyz)

        phi = math.atan2(
            xq * zq - wq * yq,
            yq * zq + wq * xq,
        )
        phi = wrap_to_pi(phi + math.radians(phi_offset_deg))

        theta = math.acos(clamp(2.0 * wq * wq - 1.0 + 2.0 * zq * zq, -1.0, 1.0))

        phi_valid = theta > math.radians(phi_valid_thresh_deg)

        return {
            "phi_rad": float(phi),
            "phi_deg": float(math.degrees(phi)),
            "theta_rad": float(theta),
            "theta_deg": float(math.degrees(theta)),
            "phi_valid": bool(phi_valid),
        }

    def get_state(self, phi_offset_deg=0.0, phi_valid_thresh_deg=3.0):
        rel = self.get_relative_arm_quaternion()
        q_rel = rel["q_rel_wxyz"]

        pcc = self.pcc_from_quaternion(
            q_rel,
            phi_offset_deg=phi_offset_deg,
            phi_valid_thresh_deg=phi_valid_thresh_deg,
        )

        return {
            "q_raw_wxyz": rel["q_raw_wxyz"],
            "q_rel_wxyz": rel["q_rel_wxyz"],
            "phi_rad": pcc["phi_rad"],
            "phi_deg": pcc["phi_deg"],
            "theta_rad": pcc["theta_rad"],
            "theta_deg": pcc["theta_deg"],
            "phi_valid": pcc["phi_valid"],
        }

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()