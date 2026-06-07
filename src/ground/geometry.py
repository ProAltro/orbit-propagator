import numpy as np

from ..constants import SPEED_OF_LIGHT
from ..maths.maths import eci_to_ecef


def elevation_azimuth_range(position_eci_km, ground_station, time_s):
    sat_ecef = eci_to_ecef(np.asarray(position_eci_km, dtype=float), time_s)
    rho = sat_ecef - ground_station.position_ecef_km
    range_km = float(np.linalg.norm(rho))
    if range_km == 0.0:
        return np.pi / 2.0, 0.0, 0.0

    lat = np.radians(ground_station.latitude_deg)
    lon = np.radians(ground_station.longitude_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array(
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
    )
    up = ground_station.local_vertical_ecef

    east_component = float(np.dot(rho, east))
    north_component = float(np.dot(rho, north))
    up_component = float(np.dot(rho, up))

    elevation = float(np.arcsin(np.clip(up_component / range_km, -1.0, 1.0)))
    azimuth = float(np.arctan2(east_component, north_component) % (2.0 * np.pi))
    return elevation, azimuth, range_km


def doppler_shift_hz(range_rate_m_s, frequency_hz):
    return -float(frequency_hz) * float(range_rate_m_s) / SPEED_OF_LIGHT
