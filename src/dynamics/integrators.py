import numpy as np

from .providers import DefaultDynamicsModel
from .state import DynamicsState
from ..maths.rotations import Omega


def _resolve_model(dynamics_model=None):
    return dynamics_model if dynamics_model is not None else DefaultDynamicsModel()


def position_ode(t, state, sat, quaternion=None, dynamics_model=None):
    q = sat.quaternion if quaternion is None else quaternion
    dynamics_state = DynamicsState(
        position_km=np.asarray(state[:3], dtype=float),
        velocity_km_s=np.asarray(state[3:6], dtype=float),
        quaternion=np.asarray(q, dtype=float),
        omega_rad_s=np.asarray(sat.omega, dtype=float),
    )
    acceleration = _resolve_model(dynamics_model).acceleration_eci_km_s2(
        t,
        dynamics_state,
        sat,
    )
    return np.concatenate((dynamics_state.velocity_km_s, acceleration))


def attitude_ode(t, state, sat, position=None, velocity=None, dt=0.0, dynamics_model=None):
    q = state[:4]
    w = state[4:]
    q_dot = 0.5 * Omega(w).dot(q)

    position_km = sat.position if position is None else position
    velocity_km_s = sat.velocity if velocity is None else velocity
    dynamics_state = DynamicsState(
        position_km=np.asarray(position_km, dtype=float),
        velocity_km_s=np.asarray(velocity_km_s, dtype=float),
        quaternion=np.asarray(q, dtype=float),
        omega_rad_s=np.asarray(w, dtype=float),
    )
    Torque = _resolve_model(dynamics_model).torque_body_n_m(t, dynamics_state, sat, dt)

    w_dot = sat.J_inv.dot(Torque - np.cross(w, np.dot(sat.J, w)))
    return np.concatenate([q_dot, w_dot])


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * k1 * dt)
    k3 = f(t + 0.5 * dt, y + 0.5 * k2 * dt)
    k4 = f(t + dt, y + k3 * dt)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def combined_ode(t, state, sat, dynamics_model=None):
    pos_state = state[:6]
    q = state[6:10]
    w = state[10:]

    pos_dot = position_ode(t, pos_state, sat, quaternion=q, dynamics_model=dynamics_model)
    att_dot = attitude_ode(
        t,
        np.concatenate((q, w)),
        sat,
        position=pos_state[:3],
        velocity=pos_state[3:6],
        dt=0.0,
        dynamics_model=dynamics_model,
    )
    return np.concatenate((pos_dot, att_dot))


def combined_rk4_step(t, sat, dt, dynamics_model=None):
    model = _resolve_model(dynamics_model)
    prepare_step = getattr(model, "prepare_step", None)
    if prepare_step is not None:
        prepare_step(t, DynamicsState.from_satellite(sat), sat, dt)
    y0 = np.concatenate((sat.position, sat.velocity, sat.quaternion, sat.omega))
    y = rk4_step(lambda t_, y_: combined_ode(t_, y_, sat, model), t, y0, dt)
    y[6:10] /= np.linalg.norm(y[6:10])
    return y


def position_rk4_step(t, sat, dt, dynamics_model=None):
    model = _resolve_model(dynamics_model)
    f = lambda t_, y_: position_ode(t_, y_, sat, dynamics_model=model)
    y = np.concatenate((sat.position, sat.velocity))
    return rk4_step(f, t, y, dt)


def attitude_rk4_step(t, sat, dt, dynamics_model=None):
    model = _resolve_model(dynamics_model)
    prepare_step = getattr(model, "prepare_step", None)
    if prepare_step is not None:
        prepare_step(t, DynamicsState.from_satellite(sat), sat, dt)
    f = lambda t_, y_: attitude_ode(t_, y_, sat, dynamics_model=model)
    y = rk4_step(f, t, np.concatenate((sat.quaternion, sat.omega)), dt)
    y[:4] /= np.linalg.norm(y[:4])
    return y
