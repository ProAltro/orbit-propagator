import numpy as np


class TimeLimitStop:
    def __init__(self, total_time_s):
        self.total_time_s = float(total_time_s)
        self._start_time_s = None

    def __call__(self, sat):
        if self._start_time_s is None:
            self._start_time_s = float(sat.time)
        return float(sat.time) - self._start_time_s >= self.total_time_s


class AngularRateBelowStop:
    def __init__(self, threshold_rad_s):
        self.threshold_rad_s = float(threshold_rad_s)

    def __call__(self, sat):
        return float(np.linalg.norm(sat.omega)) <= self.threshold_rad_s
