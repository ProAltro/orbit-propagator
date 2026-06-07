import numpy as np

from .base import Component


class Magnetorquer(Component):
    def __init__(
        self,
        max_dipole_a_m2=7.8e-4,
        bdot_gain=1.0,
        enabled=True,
        name="magnetorquer",
    ):
        super().__init__(name)
        self.max_dipole_a_m2 = float(max_dipole_a_m2)
        self.bdot_gain = float(bdot_gain)
        self.enabled = bool(enabled)
        self.previous_field_body_t = None
        self.last_dipole_a_m2 = np.zeros(3)
        self.last_torque_n_m = np.zeros(3)

    def commanded_dipole(self, magnetic_field_body_t, dt_s):
        field = np.asarray(magnetic_field_body_t, dtype=float)
        if not self.enabled or self.previous_field_body_t is None or dt_s <= 0.0:
            self.previous_field_body_t = field.copy()
            self.last_dipole_a_m2 = np.zeros(3)
            return self.last_dipole_a_m2

        b_dot = (field - self.previous_field_body_t) / dt_s
        self.previous_field_body_t = field.copy()
        norm = np.linalg.norm(b_dot)
        if norm == 0.0:
            self.last_dipole_a_m2 = np.zeros(3)
            return self.last_dipole_a_m2

        dipole = self.bdot_gain * b_dot
        dipole_norm = np.linalg.norm(dipole)
        if dipole_norm > self.max_dipole_a_m2:
            dipole = dipole / dipole_norm * self.max_dipole_a_m2
        self.last_dipole_a_m2 = dipole
        return dipole
