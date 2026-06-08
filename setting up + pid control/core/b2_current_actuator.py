import numpy as np


class CurrentTensionMapper:
    """
    Experimental mapping from identification:
        I[mA] = 72.3 * T[N] - 6.0

    Equivalent torque form with r_pulley = 0.009 m:
        I[mA] = 8029.5 * tau[Nm] - 6.0
    """

    def __init__(
        self,
        r_pulley_m=0.009,
        tension_to_current_gain_mA_per_N=72.3,
        tension_to_current_offset_mA=-6.0,
        tighten_sign=(-1.0, -1.0, -1.0),
        i_max_mA=200.0,
    ):
        self.r_pulley_m = float(r_pulley_m)
        self.k_I_per_T = float(tension_to_current_gain_mA_per_N)
        self.b_I = float(tension_to_current_offset_mA)
        self.tighten_sign = np.asarray(tighten_sign, dtype=float)
        self.i_max_mA = float(i_max_mA)

    def tension_to_torque_nm(self, tension_N):
        return float(tension_N) * self.r_pulley_m

    def torque_to_tension_N(self, tau_nm):
        return float(tau_nm) / self.r_pulley_m

    def tension_to_current_mA(self, tension_N):
        tension_N = max(0.0, float(tension_N))
        i_mA = self.k_I_per_T * tension_N + self.b_I
        return max(0.0, i_mA)

    def torque_to_current_mA(self, tau_nm):
        tension_N = self.torque_to_tension_N(tau_nm)
        return self.tension_to_current_mA(tension_N)

    def tensions_to_signed_currents_mA(self, tensions_N):
        tensions_N = np.asarray(tensions_N, dtype=float)
        mags = np.array([self.tension_to_current_mA(t) for t in tensions_N], dtype=float)
        mags = np.clip(mags, 0.0, self.i_max_mA)
        return self.tighten_sign[:len(mags)] * mags

    def current_to_tension_N(self, current_mA, motor_index):
        sign = self.tighten_sign[motor_index]
        i_tight_mA = max(0.0, sign * float(current_mA))
        tension_N = (i_tight_mA - self.b_I) / self.k_I_per_T
        return max(0.0, tension_N)

    def current_to_torque_nm(self, current_mA, motor_index):
        tension_N = self.current_to_tension_N(current_mA, motor_index)
        return self.tension_to_torque_nm(tension_N)