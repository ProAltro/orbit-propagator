import numpy as np
import json
from ..dynamics.ode_solvers import position_rk4_step, attitude_rk4_step


"""
Conventions:

- Inertia tensor is in body frame
- Position and velocity are in ECI frame
- Quaternion represents rotation from ECI to body
"""


class Satellite:
    def __init__(self):
        # altitude and velocity (ECI frame)
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)

        # attitude variables
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # (ECI to body)
        self.omega = np.zeros(3)  # rad/s (body frame)

        self.mass = 1.0
        self.drag_coefficient = 2.2
        self.J = np.diag([0.0027, 0.0027, 0.0054])
        self.J_inv = np.linalg.inv(self.J)

        # 3D structure (in body frame)
        self.n = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )
        self.A = np.array([0.01, 0.01, 0.01, 0.01, 0.1, 0.1])  # m^2
        self.solarp_area = np.array([0.006, 0.006, 0.006, 0.006, 0.006, 0.006])  # m^2
        self.drag_coefficient = 2.2  # TODO fix to actual value
        self.time = 0.0

        # TODO antenna parameters
        """
        Define self.antenna_direction_body, self.antenna_beamwidth, self.antenna_gain
        Gain distribution exactly how needs to be thought out
        Can also do Data Rate calculations later
        """

    def propagate(self, dt: float):
        pos = position_rk4_step(self.time, self, dt)
        att = attitude_rk4_step(self.time, self, dt)

        self.position = pos[:3]
        self.velocity = pos[3:6]

        self.quaternion = att[:4]
        self.omega = att[4:]

        self.time += dt

    def load_from_file(self, fp):
        """Load initial conditions from a json file"""
        with open(fp, "r") as f:
            data = json.load(f)

        self.position = np.array(data["position"])
        self.velocity = np.array(data["velocity"])
        self.quaternion = np.array(data["quaternion"])
        self.omega = np.array(data["angular_velocity"])
        self.mass = data["mass"]
