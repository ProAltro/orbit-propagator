import numpy as np
from pathlib import Path

from ..constants import EARTH_RADIUS
from ..maths.maths import A_from_q

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "atmosphere.csv"
TABLE = np.genfromtxt(DATA_PATH, delimiter=",", names=True)
ALTS = TABLE["altitude_m"]
RHOS = TABLE["density_kg_m3"]
HS = TABLE["scale_height_m"]

R_EARTH = EARTH_RADIUS * 1_000.0


def density_lookup(position_km: np.ndarray) -> float:
    """Compute atmospheric density (kg/m^3) based on tabulated exponential model."""

    altitude_m = np.linalg.norm(position_km) * 1_000.0 - R_EARTH
    if altitude_m < ALTS[0]:
        return RHOS[0]
    if altitude_m > ALTS[-1]:
        return 0.0

    for idx in range(len(ALTS) - 1):
        if altitude_m < ALTS[idx + 1]:
            break

    base_alt = ALTS[idx]
    rho0 = RHOS[idx]
    scale_height = HS[idx]
    return rho0 * np.exp(-(altitude_m - base_alt) / scale_height)


def acc_drag(position_km: np.ndarray, velocity_km_s: np.ndarray, sat) -> np.ndarray:
    """Compute atmospheric drag acceleration (km/s^2)."""

    rho = density_lookup(position_km)
    if rho <= 0.0:
        return np.zeros(3)

    OMEGA_EARTH = np.array([0.0, 0.0, 7.2921159e-5])

    # Convert position, velocity to meters
    r_m = position_km * 1_000.0
    v_m_s = velocity_km_s * 1_000.0

    # Atmospheric co-rotation velocity
    v_atm = np.cross(OMEGA_EARTH, r_m)

    # Relative velocity in ECI
    v_rel_eci = v_m_s - v_atm
    speed = np.linalg.norm(v_rel_eci)

    if speed == 0.0:
        return np.zeros(3)

    # Transform relative velocity to body frame
    A = A_from_q(sat.quaternion)  # ECI -> body
    v_rel_body = A @ v_rel_eci
    v_hat_body = v_rel_body / np.linalg.norm(v_rel_body)

    # Effective cross-sectional area
    n_hat = sat.n
    proj = n_hat @ (-v_hat_body)
    mask = proj > 0.0
    A_eff = np.sum(proj[mask] * sat.A[mask])

    if A_eff <= 0.0:
        return np.zeros(3)

    # Drag force (ECI frame)
    F_drag_eci = -0.5 * rho * sat.drag_coefficient * A_eff * speed * v_rel_eci

    # Acceleration in km/s^2
    a_drag_km_s2 = F_drag_eci / sat.mass / 1_000.0
    return a_drag_km_s2
