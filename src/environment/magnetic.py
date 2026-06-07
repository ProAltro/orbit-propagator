import numpy as np

from ..constants import EARTH_RADIUS
from ..maths.maths import ecef_to_eci

MAGNETIC_FIELD_EQUATOR_T = 3.12e-5
MAG_POLE_LAT_DEG = -80.65
MAG_POLE_LON_DEG = 107.32


def _dipole_axis_ecef() -> np.ndarray:
    lat = np.radians(MAG_POLE_LAT_DEG)
    lon = np.radians(MAG_POLE_LON_DEG)
    axis = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return axis / np.linalg.norm(axis)


def earth_magnetic_field_eci(
    position_km: np.ndarray,
    time_s: float,
    epoch_jd: float | None = None,
) -> np.ndarray:
    """Tilted-dipole geomagnetic field in ECI, returned in tesla.

    epoch_jd is accepted for interface compatibility with future IGRF-style
    backends. The current model follows the repo's numeric-second frame convention.
    """
    position = np.asarray(position_km, dtype=float)
    radius = np.linalg.norm(position)
    if radius == 0.0:
        return np.zeros(3)

    dipole_axis_eci = ecef_to_eci(_dipole_axis_ecef(), time_s)
    r_hat = position / radius
    scale = MAGNETIC_FIELD_EQUATOR_T * (EARTH_RADIUS / radius) ** 3
    field = scale * (3.0 * np.dot(dipole_axis_eci, r_hat) * r_hat - dipole_axis_eci)
    return field
