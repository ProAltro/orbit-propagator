import numpy as np

from ..constants import EARTH_RADIUS, OMEGA_EARTH
from .rotations import rot_z


def greenwich_sidereal_angle(time_s: float, epoch_jd: float | None = None) -> float:
    """Return the Earth rotation angle used by numeric-second frame conversions.

    epoch_jd is accepted as metadata for future higher-fidelity models. The current
    implementation intentionally keeps the previous repository convention: t=0 has
    ECI and ECEF aligned.
    """
    return OMEGA_EARTH * float(time_s)


def eci_to_ecef(position_eci_km: np.ndarray, time_s: float) -> np.ndarray:
    return rot_z(-greenwich_sidereal_angle(time_s)) @ np.asarray(
        position_eci_km, dtype=float
    )


def ecef_to_eci(position_ecef_km: np.ndarray, time_s: float) -> np.ndarray:
    return rot_z(greenwich_sidereal_angle(time_s)) @ np.asarray(
        position_ecef_km, dtype=float
    )


def geodetic_to_ecef(
    latitude_rad: float,
    longitude_rad: float,
    altitude_km: float = 0.0,
    radius_km: float = EARTH_RADIUS,
) -> np.ndarray:
    r = radius_km + altitude_km
    cos_lat = np.cos(latitude_rad)
    return np.array(
        [
            r * cos_lat * np.cos(longitude_rad),
            r * cos_lat * np.sin(longitude_rad),
            r * np.sin(latitude_rad),
        ]
    )


def ecef_to_geodetic(position_ecef_km: np.ndarray) -> tuple[float, float, float]:
    position = np.asarray(position_ecef_km, dtype=float)
    radius = np.linalg.norm(position)
    if radius == 0.0:
        raise ValueError("ECEF position must be non-zero.")

    latitude = np.arcsin(np.clip(position[2] / radius, -1.0, 1.0))
    longitude = np.arctan2(position[1], position[0])
    altitude = radius - EARTH_RADIUS
    return float(latitude), float(longitude), float(altitude)


def eci_to_geodetic(position_eci_km: np.ndarray, time_s: float) -> tuple[float, float, float]:
    return ecef_to_geodetic(eci_to_ecef(position_eci_km, time_s))
