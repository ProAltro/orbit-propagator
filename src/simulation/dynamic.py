from ..dynamics.ode_solvers import combined_rk4_step
from ..dynamics.providers import DefaultDynamicsModel
from .runner import SimulationRunner


class DynamicSimulation(SimulationRunner):
    def __init__(self, sat, propagator=None):
        super().__init__(sat)
        self.propagator = propagator if propagator is not None else DefaultDynamicsModel()

    def step(self, dt_s):
        state = combined_rk4_step(
            self.sat.time,
            self.sat,
            dt_s,
            dynamics_model=self.propagator,
        )
        self.sat.position = state[:3]
        self.sat.velocity = state[3:6]
        self.sat.quaternion = state[6:10]
        self.sat.omega = state[10:]
        self.sat.time += dt_s
        self.sample_monitors()
        return self.sat
