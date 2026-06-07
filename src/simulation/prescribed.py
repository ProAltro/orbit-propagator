import numpy as np

from ..maths.maths import Omega
from .runner import SimulationRunner


class PrescribedOrbitSimulation(SimulationRunner):
    def __init__(self, sat, orbit):
        super().__init__(sat)
        self.orbit = orbit

    def step(self, dt_s):
        next_time = self.sat.time + dt_s
        self.sat.position = self.orbit.position_at(next_time)
        if hasattr(self.orbit, "velocity_at"):
            self.sat.velocity = self.orbit.velocity_at(next_time)

        q_dot = 0.5 * Omega(self.sat.omega) @ self.sat.quaternion
        self.sat.quaternion = self.sat.quaternion + q_dot * dt_s
        self.sat.quaternion = self.sat.quaternion / np.linalg.norm(self.sat.quaternion)
        self.sat.time = next_time
        self.sample_monitors()
        return self.sat
