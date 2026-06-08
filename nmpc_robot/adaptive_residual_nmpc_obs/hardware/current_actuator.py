import numpy as np


class CurrentTensionMapper:
    """
    Convert tendon tension [N] to motor current [mA].
    Also convert measured current back to estimated tension.
    """

    def __init__(self, gain_mA_per_N, offset_mA, tighten_sign, i_max_mA):
        self.gain_mA_per_N = float(gain_mA_per_N)
        self.offset_mA = float(offset_mA)
        self.tighten_sign = np.asarray(tighten_sign, dtype=float)
        self.i_max_mA = float(i_max_mA)

    def tension_to_current_mA(self, tension_N):
        tension_N = max(0.0, float(tension_N))
        if tension_N <= 1e-12:
            return 0.0

        current_mA = self.gain_mA_per_N * tension_N + self.offset_mA
        return max(0.0, current_mA)

    def tensions_to_signed_currents_mA(self, tensions_N):
        tensions_N = np.asarray(tensions_N, dtype=float)

        current_abs = np.array(
            [self.tension_to_current_mA(t) for t in tensions_N],
            dtype=float,
        )

        current_abs = np.clip(current_abs, 0.0, self.i_max_mA)
        return self.tighten_sign[:len(current_abs)] * current_abs

    def measured_current_to_tension_N(self, current_mA, motor_index):
        sign = self.tighten_sign[motor_index]
        current_tight_mA = max(0.0, sign * float(current_mA))

        if current_tight_mA <= self.offset_mA:
            return 0.0

        tension_N = (current_tight_mA - self.offset_mA) / self.gain_mA_per_N
        return max(0.0, tension_N)