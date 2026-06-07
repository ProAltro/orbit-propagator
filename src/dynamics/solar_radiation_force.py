import numpy as np

from ..constants import (
    ASTRONOMICAL_UNIT_KM,
    EARTH_RADIUS,
    SOLAR_FLUX,
    SPEED_OF_LIGHT,
)
from ..environment.sun import sun_direction_unit
from ..maths.maths import A_from_q


def is_cylindrical_eclipse(position_km, sun_direction_eci, earth_radius_km=EARTH_RADIUS):
    position = np.asarray(position_km, dtype=float)
    sun_dir = np.asarray(sun_direction_eci, dtype=float)
    sun_norm = np.linalg.norm(sun_dir)
    if sun_norm == 0.0:
        return False

    sun_dir = sun_dir / sun_norm
    projection = np.dot(position, sun_dir)
    perpendicular = position - projection * sun_dir
    return bool(projection < 0.0 and np.linalg.norm(perpendicular) < earth_radius_km)


def solar_radiation_acceleration(time_s, position_km, sat, quaternion=None):
    """Surface-model SRP acceleration in km/s^2."""
    surface = sat.get_component("body_surfaces", default=None)
    if surface is None:
        return np.zeros(3)

    epoch_jd = getattr(sat, "epoch_julian_date", None)
    if epoch_jd is None:
        sun_dir_eci = sun_direction_unit(time_s)
    else:
        sun_dir_eci = sun_direction_unit(time_s, epoch_jd=epoch_jd)

    if is_cylindrical_eclipse(position_km, sun_dir_eci):
        return np.zeros(3)

    q = sat.quaternion if quaternion is None else quaternion
    body_to_eci = A_from_q(q).T
    sat_to_sun = ASTRONOMICAL_UNIT_KM * sun_dir_eci - np.asarray(position_km, dtype=float)
    sat_to_sun_norm = np.linalg.norm(sat_to_sun)
    if sat_to_sun_norm == 0.0:
        return np.zeros(3)
    sat_to_sun_unit = sat_to_sun / sat_to_sun_norm

    pressure_pa = SOLAR_FLUX / SPEED_OF_LIGHT
    force_n = np.zeros(3)
    for normal_body, area_m2, specular, diffuse in zip(
        surface.normals_body,
        surface.srp_areas_m2,
        surface.specular,
        surface.diffuse,
    ):
        normal_eci = body_to_eci @ normal_body
        incidence = float(np.dot(normal_eci, sat_to_sun_unit))
        if incidence <= 0.0 or area_m2 <= 0.0:
            continue

        reflection = (
            (1.0 - specular) * sat_to_sun_unit
            + 2.0 * (specular * incidence + diffuse / 3.0) * normal_eci
        )
        force_n += -pressure_pa * area_m2 * incidence * reflection

    if sat.mass <= 0.0:
        raise ValueError("Satellite mass must be positive for SRP acceleration.")
    return force_n / sat.mass / 1000.0


def solar_radiation_force(position_km, sat, time_s, quaternion=None):
    """Backward-compatible name returning acceleration, not force."""
    return solar_radiation_acceleration(time_s, position_km, sat, quaternion=quaternion)
