import numpy as np

from ..external.environment.magnetic import earth_magnetic_field_eci
from ..maths.rotations import A_from_q
from ..satellite.components.actuators import Magnetorquer


def magnetic_field_body(position_km, time_s, quaternion, sat):
    epoch_jd = getattr(sat, "epoch_julian_date", None)
    field_eci = earth_magnetic_field_eci(position_km, time_s, epoch_jd=epoch_jd)
    return A_from_q(quaternion) @ field_eci


def magnetorquer_torque_body(
    position_km,
    time_s,
    quaternion,
    sat,
    dt_s=0.0,
):
    if dt_s <= 0.0:
        return last_magnetorquer_torque_body(sat)

    field_body = magnetic_field_body(position_km, time_s, quaternion, sat)
    torque = np.zeros(3)
    for magnetorquer in sat.get_components(Magnetorquer):
        dipole = magnetorquer.commanded_dipole(field_body, dt_s)
        component_torque = np.cross(dipole, field_body)
        magnetorquer.last_torque_n_m = component_torque
        torque += component_torque
    return torque


def last_magnetorquer_torque_body(sat):
    torque = np.zeros(3)
    for magnetorquer in sat.get_components(Magnetorquer):
        torque += magnetorquer.last_torque_n_m
    return torque
