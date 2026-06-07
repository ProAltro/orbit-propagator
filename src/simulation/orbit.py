import numpy as np


class CircularOrbit:
    def __init__(
        self,
        initial_position,
        initial_velocity,
    ):
        self.initial_position = np.asarray(initial_position, dtype=float)
        self.initial_velocity = np.asarray(initial_velocity, dtype=float)

        self.radius_km = np.linalg.norm(self.initial_position)
        self.speed_km_s = np.linalg.norm(self.initial_velocity)
        if self.radius_km == 0.0:
            raise ValueError("initial_position must be non-zero.")

        h = np.cross(self.initial_position, self.initial_velocity)
        h_norm = np.linalg.norm(h)
        if h_norm == 0.0:
            raise ValueError("initial_position and initial_velocity cannot be parallel.")

        self.mean_motion_rad_s = self.speed_km_s / self.radius_km
        self._r_hat = self.initial_position / self.radius_km
        self._h_hat = h / h_norm
        self._v_hat = np.cross(self._h_hat, self._r_hat)

    def position_at(self, time_s):
        theta = self.mean_motion_rad_s * time_s
        return self.radius_km * (
            np.cos(theta) * self._r_hat + np.sin(theta) * self._v_hat
        )

    def velocity_at(self, time_s):
        theta = self.mean_motion_rad_s * time_s
        return self.speed_km_s * (
            -np.sin(theta) * self._r_hat + np.cos(theta) * self._v_hat
        )

    @classmethod
    def from_satellite(cls, sat):
        return cls(sat.position, sat.velocity)

    def simulate(self, sat, total_time, dt):
        from .prescribed import PrescribedOrbitSimulation

        sat.position = self.initial_position.copy()
        sat.velocity = self.initial_velocity.copy()
        return PrescribedOrbitSimulation(sat, self).run(total_time, dt)


def simulate_circular_orbit(
    sat,
    total_time,
    dt,
):
    orbit = CircularOrbit.from_satellite(sat)
    return orbit.simulate(sat, total_time, dt)
