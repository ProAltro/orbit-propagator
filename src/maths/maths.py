import numpy as np


def cross_product_matrix(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def A_from_q(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    q /= norm

    w = q[0]
    v = q[1:]

    return (
        np.eye(3) * (w * w - np.dot(v, v))
        + 2.0 * np.outer(v, v)
        + 2.0 * w * cross_product_matrix(v)
    )


def euler_from_A(A: np.ndarray) -> tuple[np.ndarray, float]:
    trace = np.trace(A)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)

    if np.isclose(angle, 0.0):
        return np.array([1.0, 0.0, 0.0]), 0.0

    axis = np.array([A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]])
    axis /= 2.0 * np.sin(angle)
    axis /= np.linalg.norm(axis)
    return axis, angle


def q_from_euler(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    axis /= norm

    half_angle = angle / 2.0
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    return np.array([w, xyz[0], xyz[1], xyz[2]])


def q_from_A(A: np.ndarray) -> np.ndarray:
    return q_from_euler(*euler_from_A(A))


def Omega(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array(
        [
            [0, -wx, -wy, -wz],
            [wx, 0, -wz, wy],
            [wy, wz, 0, -wx],
            [wz, -wy, wx, 0],
        ]
    )


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
