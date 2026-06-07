from ..utils.initial_conditions import keplerian_to_state_vectors
from .orbit import CircularOrbit


class KeplerianCircularOrbit(CircularOrbit):
    def __init__(
        self,
        altitude_km,
        inclination_deg,
        raan_deg=0.0,
        argument_of_perigee_deg=0.0,
        true_anomaly_deg=0.0,
    ):
        position, velocity = keplerian_to_state_vectors(
            altitude_km,
            inclination_deg,
            raan_deg=raan_deg,
            argument_of_perigee_deg=argument_of_perigee_deg,
            true_anomaly_deg=true_anomaly_deg,
        )
        super().__init__(position, velocity)
        self.altitude_km = float(altitude_km)
        self.inclination_deg = float(inclination_deg)
        self.raan_deg = float(raan_deg)
        self.argument_of_perigee_deg = float(argument_of_perigee_deg)
        self.true_anomaly_deg = float(true_anomaly_deg)
