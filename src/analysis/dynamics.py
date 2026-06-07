import numpy as np

from .monitor import AnalysisMonitor
from ..dynamics.magnetic_torque import magnetic_field_body
from ..dynamics.solar_radiation_force import solar_radiation_acceleration


class DynamicsEnvironmentMonitor(AnalysisMonitor):
    def __init__(self, name="dynamics_environment"):
        super().__init__(name)
        self.times = []
        self.magnetic_fields_body_t = []
        self.magnetorquer_torques_body_n_m = []
        self.srp_accelerations_km_s2 = []

    def sample(self, sat):
        self.times.append(float(sat.time))
        self.magnetic_fields_body_t.append(
            magnetic_field_body(sat.position, sat.time, sat.quaternion, sat)
        )
        torque = np.zeros(3)
        for component in sat.components:
            torque += getattr(component, "last_torque_n_m", np.zeros(3))
        self.magnetorquer_torques_body_n_m.append(torque)
        self.srp_accelerations_km_s2.append(
            solar_radiation_acceleration(sat.time, sat.position, sat)
        )
