import numpy as np

from ...constants import EARTH_RADIUS
from ...maths.frames import ecef_to_eci, geodetic_to_ecef


class GroundStation:
    def __init__(
        self,
        latitude_deg: float,
        longitude_deg: float,
        altitude_km: float = 0.0,
        radius_km: float = EARTH_RADIUS,
    ):
        self.latitude_deg = float(latitude_deg)
        self.longitude_deg = float(longitude_deg)
        self.altitude_km = float(altitude_km)
        self.radius_km = float(radius_km)

        lat = np.radians(self.latitude_deg)
        lon = np.radians(self.longitude_deg)
        self.position_ecef_km = geodetic_to_ecef(
            lat,
            lon,
            altitude_km=self.altitude_km,
            radius_km=self.radius_km,
        )
        self.local_vertical_ecef = self.position_ecef_km / np.linalg.norm(
            self.position_ecef_km
        )

    def position_eci_km(self, time_s):
        return ecef_to_eci(self.position_ecef_km, time_s)
