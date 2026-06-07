from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicsState:
    position_km: np.ndarray
    velocity_km_s: np.ndarray
    quaternion: np.ndarray
    omega_rad_s: np.ndarray

    @classmethod
    def from_vector(cls, vector):
        return cls(
            position_km=np.asarray(vector[:3], dtype=float),
            velocity_km_s=np.asarray(vector[3:6], dtype=float),
            quaternion=np.asarray(vector[6:10], dtype=float),
            omega_rad_s=np.asarray(vector[10:], dtype=float),
        )

    @classmethod
    def from_satellite(cls, sat):
        return cls(
            position_km=np.asarray(sat.position, dtype=float),
            velocity_km_s=np.asarray(sat.velocity, dtype=float),
            quaternion=np.asarray(sat.quaternion, dtype=float),
            omega_rad_s=np.asarray(sat.omega, dtype=float),
        )

    def as_vector(self):
        return np.concatenate(
            (
                self.position_km,
                self.velocity_km_s,
                self.quaternion,
                self.omega_rad_s,
            )
        )
