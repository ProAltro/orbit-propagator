import numpy as np

from ...constants import JULIAN_DATE_J2000


def sun_direction_unit(
    time_s: float,
    epoch_jd: float = JULIAN_DATE_J2000,
) -> np.ndarray:
    """Approximate inertial Earth-to-Sun unit vector.

    This is the compact solar-position approximation from the update branch,
    decoupled from Satellite and driven by numeric seconds plus optional epoch.
    """
    jd = float(epoch_jd) + float(time_s) / 86400.0
    centuries = (jd - JULIAN_DATE_J2000) / 36525.0

    mean_longitude = np.radians(280.460 + 36000.771 * centuries)
    mean_anomaly = np.radians(357.5277233 + 35999.05034 * centuries)
    ecliptic_longitude = (
        mean_longitude
        + np.radians(1.914666471) * np.sin(mean_anomaly)
        + np.radians(0.019994643) * np.sin(2.0 * mean_anomaly)
    )
    obliquity = np.radians(23.439291 - 0.0130042 * centuries)

    direction = np.array(
        [
            np.cos(ecliptic_longitude),
            np.cos(obliquity) * np.sin(ecliptic_longitude),
            np.sin(obliquity) * np.sin(ecliptic_longitude),
        ]
    )
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0])
    return direction / norm
