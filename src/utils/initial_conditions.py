import numpy as np

from ..constants import EARTH_MU, EARTH_RADIUS
from ..maths.rotations import rot_x, rot_z


def keplerian_to_state_vectors(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float = 0.0,
    argument_of_perigee_deg: float = 0.0,
    true_anomaly_deg: float = 0.0,
    mu_km3_s2: float = EARTH_MU,
    earth_radius_km: float = EARTH_RADIUS,
):
    """Circular Keplerian elements to ECI position/velocity in km and km/s."""
    radius_km = earth_radius_km + float(altitude_km)
    inclination = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    argument_of_perigee = np.radians(argument_of_perigee_deg)
    true_anomaly = np.radians(true_anomaly_deg)
    argument_of_latitude = argument_of_perigee + true_anomaly

    r_perifocal = radius_km * np.array(
        [np.cos(argument_of_latitude), np.sin(argument_of_latitude), 0.0]
    )
    speed = np.sqrt(mu_km3_s2 / radius_km)
    v_perifocal = speed * np.array(
        [-np.sin(argument_of_latitude), np.cos(argument_of_latitude), 0.0]
    )

    rotation = rot_z(raan) @ rot_x(inclination)
    return rotation @ r_perifocal, rotation @ v_perifocal


def generate_inclined_circular_orbit(
    inclination_deg: float,
    radius_km: float,
    raan_deg: float = None,
    arg_lat_deg: float = None,
    rng: np.random.Generator = None,
    mu_km3_s2: float = EARTH_MU,
):
    if rng is None:
        rng = np.random.default_rng()

    if raan_deg is None:
        raan_deg = rng.uniform(0.0, 360.0)

    if arg_lat_deg is None:
        arg_lat_deg = rng.uniform(0.0, 360.0)

    inc = np.radians(inclination_deg)
    raan = np.radians(raan_deg)
    arg_lat = np.radians(arg_lat_deg)

    r_orbital = radius_km * np.array([np.cos(arg_lat), np.sin(arg_lat), 0.0])
    v_mag = np.sqrt(mu_km3_s2 / radius_km)
    v_orbital = v_mag * np.array([-np.sin(arg_lat), np.cos(arg_lat), 0.0])

    r_raan = np.array(
        [
            [np.cos(raan), -np.sin(raan), 0.0],
            [np.sin(raan), np.cos(raan), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    r_inc = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(inc), -np.sin(inc)],
            [0.0, np.sin(inc), np.cos(inc)],
        ]
    )

    rotation = r_raan @ r_inc
    return rotation @ r_orbital, rotation @ v_orbital


def random_quaternion(rng: np.random.Generator = None):
    if rng is None:
        rng = np.random.default_rng()

    q = rng.normal(size=4)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm
