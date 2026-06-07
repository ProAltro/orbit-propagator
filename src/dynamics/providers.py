from typing import Protocol

import numpy as np

from ..constants import EARTH_MU
from .atmospheric_drag import acc_drag
from .j2_acceleration import acc_j2
from .magnetic_torque import last_magnetorquer_torque_body, magnetorquer_torque_body
from .solar_radiation_force import solar_radiation_acceleration


class ForceProvider(Protocol):
    def acceleration_eci_km_s2(self, time_s, state, sat) -> np.ndarray:
        ...


class TorqueProvider(Protocol):
    def torque_body_n_m(self, time_s, state, sat, dt_s) -> np.ndarray:
        ...


class KeplerGravityProvider:
    def acceleration_eci_km_s2(self, time_s, state, sat):
        r = state.position_km
        r_norm = np.linalg.norm(r)
        if r_norm == 0.0:
            raise ValueError("Position state must be non-zero before propagation.")
        return -EARTH_MU * r / r_norm**3


class J2GravityProvider:
    def acceleration_eci_km_s2(self, time_s, state, sat):
        return acc_j2(state.position_km)


class AtmosphericDragProvider:
    def acceleration_eci_km_s2(self, time_s, state, sat):
        return acc_drag(
            state.position_km,
            state.velocity_km_s,
            sat,
            quaternion=state.quaternion,
        )


class SolarRadiationPressureProvider:
    def acceleration_eci_km_s2(self, time_s, state, sat):
        return solar_radiation_acceleration(
            time_s,
            state.position_km,
            sat,
            quaternion=state.quaternion,
        )


class MagnetorquerTorqueProvider:
    def prepare_step(self, time_s, state, sat, dt_s):
        magnetorquer_torque_body(
            state.position_km,
            time_s,
            state.quaternion,
            sat,
            dt_s,
        )

    def torque_body_n_m(self, time_s, state, sat, dt_s):
        return last_magnetorquer_torque_body(sat)


class DefaultDynamicsModel:
    def __init__(self, force_providers=None, torque_providers=None):
        self.force_providers = (
            list(force_providers)
            if force_providers is not None
            else [
                KeplerGravityProvider(),
                J2GravityProvider(),
                AtmosphericDragProvider(),
                SolarRadiationPressureProvider(),
            ]
        )
        self.torque_providers = (
            list(torque_providers)
            if torque_providers is not None
            else [MagnetorquerTorqueProvider()]
        )

    def acceleration_eci_km_s2(self, time_s, state, sat):
        acceleration = np.zeros(3)
        for provider in self.force_providers:
            acceleration += provider.acceleration_eci_km_s2(time_s, state, sat)
        return acceleration

    def prepare_step(self, time_s, state, sat, dt_s):
        for provider in self.torque_providers:
            prepare_step = getattr(provider, "prepare_step", None)
            if prepare_step is not None:
                prepare_step(time_s, state, sat, dt_s)

    def torque_body_n_m(self, time_s, state, sat, dt_s):
        torque = np.zeros(3)
        for provider in self.torque_providers:
            torque += provider.torque_body_n_m(time_s, state, sat, dt_s)
        return torque
