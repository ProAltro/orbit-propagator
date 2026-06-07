import numpy as np
import json

from ..constants import JULIAN_DATE_J2000
from ..dynamics.providers import DefaultDynamicsModel
from .components.power import SolarPanelArray
from .components.surfaces import BodySurfaceModel


"""
Conventions:

- Inertia tensor is in body frame
- Position and velocity are in ECI frame
- Quaternion represents rotation from ECI to body
"""


class Satellite:
    def __init__(self, install_default_components=True, dynamics_model=None):
        # altitude and velocity (ECI frame)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        # attitude variables
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (ECI to body)
        self.omega = np.zeros(3)  # rad/s (body frame)

        self.mass = 1.0
        self.J = np.diag([0.0027, 0.0027, 0.0054])
        self.J_inv = np.linalg.inv(self.J)

        self.time = 0.0
        self.epoch_julian_date = JULIAN_DATE_J2000
        self.epoch_datetime = None
        self.dynamics_model = (
            dynamics_model if dynamics_model is not None else DefaultDynamicsModel()
        )
        self.components = []
        self.monitors = []
        if install_default_components:
            self.add_component(BodySurfaceModel())
            self.add_component(SolarPanelArray())

        # TODO antenna parameters
        """
        Define self.antenna_direction_body, self.antenna_beamwidth, self.antenna_gain
        Gain distribution exactly how needs to be thought out
        Can also do Data Rate calculations later
        """

    @property
    def n(self):
        return self.get_component("body_surfaces").normals_body

    @property
    def A(self):
        return self.get_component("body_surfaces").drag_areas_m2

    @property
    def solarp_area(self):
        return self.get_component("solar_panels").areas_m2

    @property
    def drag_coefficient(self):
        return self.get_component("body_surfaces").drag_coefficient

    def add_component(self, component):
        name = getattr(component, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Component must define a non-empty string name.")

        if self.get_component(name, default=None) is not None:
            raise ValueError(f"Component with name {name!r} already exists.")

        if hasattr(component, "attach"):
            component.attach(self)

        self.components.append(component)
        return component

    def get_component(self, name, default=...):
        for component in self.components:
            if getattr(component, "name", None) == name:
                return component

        if default is ...:
            raise KeyError(f"No component named {name!r}.")
        return default

    def get_components(self, component_type=None):
        if component_type is None:
            return list(self.components)
        return [
            component
            for component in self.components
            if isinstance(component, component_type)
        ]

    def add_monitor(self, monitor):
        name = getattr(monitor, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("Monitor must define a non-empty string name.")
        if not callable(getattr(monitor, "sample", None)):
            raise ValueError("Monitor must define a callable sample(sat) method.")

        if self.get_monitor(name, default=None) is not None:
            raise ValueError(f"Monitor with name {name!r} already exists.")

        if hasattr(monitor, "attach"):
            monitor.attach(self)

        self.monitors.append(monitor)
        return monitor

    def get_monitor(self, name, default=...):
        for monitor in self.monitors:
            if getattr(monitor, "name", None) == name:
                return monitor

        if default is ...:
            raise KeyError(f"No monitor named {name!r}.")
        return default

    def propagate(self, dt: float):
        from ..simulation.dynamic import DynamicSimulation

        return DynamicSimulation(self, propagator=self.dynamics_model).step(dt)

    def load_from_file(self, fp):
        """Load initial conditions from a json file"""
        with open(fp, "r") as f:
            data = json.load(f)

        self.position = np.array(data["position"])
        self.velocity = np.array(data["velocity"])
        self.quaternion = np.array(data["quaternion"])
        self.omega = np.array(data["angular_velocity"])
        self.mass = data["mass"]
