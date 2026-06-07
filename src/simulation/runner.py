class SimulationRunner:
    def __init__(self, sat):
        self.sat = sat

    def step(self, dt_s):
        raise NotImplementedError

    def run(self, total_time_s, dt_s, stop_condition=None):
        steps = int(total_time_s / dt_s)
        for _ in range(steps):
            if stop_condition is not None and stop_condition(self.sat):
                break
            self.step(dt_s)
            if stop_condition is not None and stop_condition(self.sat):
                break
        return self.sat

    def sample_monitors(self):
        for monitor in self.sat.monitors:
            monitor.sample(self.sat)
