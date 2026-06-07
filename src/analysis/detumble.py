import numpy as np

from .monitor import AnalysisMonitor


class AngularRateMonitor(AnalysisMonitor):
    def __init__(self, threshold_rad_s=None, name="angular_rate"):
        super().__init__(name)
        self.threshold_rad_s = (
            None if threshold_rad_s is None else float(threshold_rad_s)
        )
        self.times = []
        self.omega_body_rad_s = []
        self.rate_norm_rad_s = []

    @property
    def detumble_time_s(self):
        if self.threshold_rad_s is None:
            return None

        for time_s, rate in zip(self.times, self.rate_norm_rad_s):
            if rate <= self.threshold_rad_s:
                return float(time_s)
        return None

    def sample(self, sat):
        omega = np.asarray(sat.omega, dtype=float).copy()
        self.times.append(float(sat.time))
        self.omega_body_rad_s.append(omega)
        self.rate_norm_rad_s.append(float(np.linalg.norm(omega)))
