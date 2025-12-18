import numpy as np


def sun_direction_unit(time: float) -> np.ndarray:
    """Placeholder: returns approximate sun direction unit vector."""
    # TODO: implement proper sun position calculation
    return np.array([1.0, 0.0, 0.0])


def solar_radiation_force(position_km: np.ndarray, sat, time: float) -> np.ndarray:
    """Placeholder: returns zero solar radiation force."""
    # TODO: implement solar radiation pressure
    return np.zeros(3)
