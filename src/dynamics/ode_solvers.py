import numpy as np

from ..constants import EARTH_MU
from .atmospheric_drag import acc_drag
from .j2_acceleration import acc_j2
from .solar_radiation_force import solar_radiation_force
from ..maths.maths import Omega


# TODO Precision Improvements (Minor Impact)
"""
When passing sat to the intermediate rk4 steps, the SRP and Drag calculations might not be updated
"""


def position_ode(t, state, sat):
    r = state[:3]
    a_kepler = -EARTH_MU * r / np.linalg.norm(r) ** 3
    a_j2 = acc_j2(r)
    # a_solar_radiation = solar_radiation_force(t, r, sat)
    a_drag = acc_drag(r, state[3:6], sat)

    a = a_kepler + a_j2 + a_drag
    return np.concatenate((state[3:6], a))


def attitude_ode(t, state, sat):
    q = state[:4]
    w = state[4:]
    q_dot = 0.5 * Omega(w).dot(q)

    # TODO External Torques (Gravity Gradient, Atmospheric, Solar Radiation)
    Torque = np.zeros(3)

    w_dot = sat.J_inv.dot(Torque - np.cross(w, np.dot(sat.J, w)))
    return np.concatenate([q_dot, w_dot])


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + 0.5 * k2 * dt)
    k4 = f(t + dt, y + k3 * dt)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def position_rk4_step(t, sat, dt):
    f = lambda t_, y_: position_ode(t_, y_, sat)
    y = np.concatenate((sat.position, sat.velocity))
    return rk4_step(f, t, y, dt)


def attitude_rk4_step(t, sat, dt):
    f = lambda t_, y_: attitude_ode(t_, y_, sat)
    y = rk4_step(f, t, np.concatenate((sat.quaternion, sat.omega)), dt)
    y[:4] /= np.linalg.norm(y[:4])
    return y
