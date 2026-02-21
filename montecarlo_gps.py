"""
Monte Carlo GPS Visibility Analysis
====================================
Ground station : 28.3802 N, 75.6092 E
Orbit          : 450 km circular, 60 deg inclination
MC parameter   : RAAN ~ Uniform[0, 360 deg]
Simulation     : 1000 independent RAAN draws, each run for ONE orbital period
Metric         : Total contact time per satellite (elevation >= 60 deg)
"""

import numpy as np
import matplotlib.pyplot as plt

from src.constants import EARTH_RADIUS, EARTH_MU, OMEGA_EARTH

# ── Configuration ─────────────────────────────────────────────────────────────
GROUND_LAT_DEG = 28.3802  # deg North
GROUND_LON_DEG = 75.6092  # deg East
ALT_KM = 450.0  # km
INC_DEG = 60.0  # deg
EL_MIN_DEG = 30.0  # deg  (minimum elevation for contact)

N_SAMPLES = 1000  # number of RAAN draws
DT = 0.1  # time step (seconds)
SEED = 42

# ── Derived constants ─────────────────────────────────────────────────────────
R_E = EARTH_RADIUS
SMA = R_E + ALT_KM

GROUND_LAT = np.deg2rad(GROUND_LAT_DEG)
GROUND_LON = np.deg2rad(GROUND_LON_DEG)
INC = np.deg2rad(INC_DEG)
EL_MIN = np.deg2rad(EL_MIN_DEG)

N_MOT = np.sqrt(EARTH_MU / SMA**3)  # rad/s
T_ORB = 2.0 * np.pi / N_MOT  # one orbital period (s)

GS_ECEF = R_E * np.array(
    [
        np.cos(GROUND_LAT) * np.cos(GROUND_LON),
        np.cos(GROUND_LAT) * np.sin(GROUND_LON),
        np.sin(GROUND_LAT),
    ]
)
GS_UNIT = GS_ECEF / np.linalg.norm(GS_ECEF)

# ── Analytical propagators ────────────────────────────────────────────────────


def orbit_eci(times: np.ndarray, raan: float) -> np.ndarray:
    """
    Closed-form ECI position (km) for a circular orbit.
    AOP = 0, TA₀ = 0  →  true anomaly = n·t

    Parameters
    ----------
    times : (N,) array of times in seconds
    raan  : RAAN in radians

    Returns
    -------
    pos_eci : (N, 3) array  [km]
    """
    ta = N_MOT * times  # true anomaly
    cos_ta, sin_ta = np.cos(ta), np.sin(ta)
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_inc, sin_inc = np.cos(INC), np.sin(INC)

    # Perifocal → ECI rotation (AOP = 0, Bate–Mueller–White §2.6)
    x = SMA * (cos_raan * cos_ta - sin_raan * sin_ta * cos_inc)
    y = SMA * (sin_raan * cos_ta + cos_raan * sin_ta * cos_inc)
    z = SMA * (sin_inc * sin_ta)

    return np.column_stack([x, y, z])


def eci_to_ecef(pos_eci: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Rotate ECI → ECEF via Earth's sidereal rotation (simplified, GAST₀ = 0).

    Parameters
    ----------
    pos_eci : (N, 3) array  [km]
    times   : (N,)   array  [seconds]

    Returns
    -------
    pos_ecef : (N, 3) array  [km]
    """
    theta = OMEGA_EARTH * times  # Earth rotation angle (rad)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    x = cos_t * pos_eci[:, 0] + sin_t * pos_eci[:, 1]
    y = -sin_t * pos_eci[:, 0] + cos_t * pos_eci[:, 1]
    z = pos_eci[:, 2]

    return np.column_stack([x, y, z])


def elevation_angle(pos_ecef: np.ndarray) -> np.ndarray:
    """
    Elevation angle (rad) of the satellite seen from the ground station.

    Uses the identity:
        el = arcsin( (ρ̂ · r̂_gs) )
    where ρ = pos_ecef − GS_ECEF and r̂_gs is the local vertical unit vector.

    Parameters
    ----------
    pos_ecef : (N, 3) array  [km]

    Returns
    -------
    el : (N,) array  [radians]
    """
    rho = pos_ecef - GS_ECEF[None, :]  # ground→sat vector
    rho_norm = np.linalg.norm(rho, axis=1, keepdims=True)
    gs_unit = GS_ECEF / np.linalg.norm(GS_ECEF)
    cos_zen = np.einsum("ij,j->i", rho / rho_norm, gs_unit)
    return np.arcsin(cos_zen)  # el = 90° − zenith


def contact_time(times: np.ndarray, raan: float) -> float:
    """Total contact time (s) over one orbital period for a given RAAN."""
    pos_eci = orbit_eci(times, raan)
    pos_ecef = eci_to_ecef(pos_eci, times)
    rho = pos_ecef - GS_ECEF[None, :]
    rho_norm = np.linalg.norm(rho, axis=1, keepdims=True)
    cos_zen = np.einsum("ij,j->i", rho / rho_norm, GS_UNIT)
    el = np.arcsin(cos_zen)
    return (el >= EL_MIN).sum() * DT


# ── Monte Carlo ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
raan_samples = rng.uniform(0.0, 2.0 * np.pi, N_SAMPLES)
times = np.arange(0.0, T_ORB, DT)  # exactly one orbital period

print(
    f"Running Monte Carlo: {N_SAMPLES} RAAN samples, "
    f"1 orbital period each ({T_ORB/60:.2f} min), dt = {DT} s ..."
)

contacts = np.array([contact_time(times, raan) for raan in raan_samples])

print("Done.\n")

# ── Statistics ────────────────────────────────────────────────────────────────
nonzero = contacts[contacts > 0]
ref_mean = nonzero.mean() if len(nonzero) else 1.0
contact_pct = (contacts > 0).mean() * 100

sep = "=" * 60
print(sep)
print("  Monte Carlo GPS Visibility  --  Overall Results")
print(sep)
print(f"  Ground station : {GROUND_LAT_DEG} N, {GROUND_LON_DEG} E")
print(f"  Orbit          : {ALT_KM:.0f} km, {INC_DEG:.0f} deg incl., circular")
print(f"  Min elevation  : {EL_MIN_DEG:.0f} deg")
print(f"  Orbital period : {T_ORB/60:.2f} min")
print(f"  Samples        : {N_SAMPLES}  |  RAAN ~ Uniform[0, 360 deg]")
print(f"  Time step      : {DT} s")
print(sep)

print(f"\n  Contact statistics across all {N_SAMPLES} RAAN samples")
print(f"  (each satellite simulated for one full orbital period)\n")

print(f"  {'Metric':<24}  {'Seconds':>10}  {'Minutes':>10}")
print(f"  {'-'*46}")
for label, val in [
    ("Min  (all samples)", contacts.min()),
    ("Mean (all samples, incl. 0)", contacts.mean()),
    ("Median (all samples)", np.median(contacts)),
    ("Max  (all samples)", contacts.max()),
    ("Mean (contact passes only)", ref_mean),
    ("Median (contact passes only)", np.median(nonzero) if len(nonzero) else 0.0),
]:
    print(f"  {label:<24}  {val:>10.1f}  {val/60:>10.3f}")

print(
    f"\n  Passes with any contact : {contact_pct:.1f}%  "
    f"({int(round(contact_pct / 100 * N_SAMPLES))} of {N_SAMPLES})"
)

print(
    f"\n  Fraction of all {N_SAMPLES} passes below X% of contact-only mean "
    f"(ref = {ref_mean:.1f} s):"
)
print(f"  {'Threshold':<16}  {'Fraction':>10}")
print(f"  {'-'*28}")
for th, lab in [
    (0.50, "< 50% of mean"),
    (0.25, "< 25% of mean"),
    (0.10, "< 10% of mean"),
]:
    frac = (contacts < th * ref_mean).mean() * 100
    print(f"  {lab:<16}  {frac:>9.1f}%")

print(f"\n{sep}\n")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"MC GPS Visibility -- {ALT_KM:.0f} km, {INC_DEG:.0f} deg incl., "
    f"El>={EL_MIN_DEG:.0f} deg  |  "
    f"GS: {GROUND_LAT_DEG}N {GROUND_LON_DEG}E  |  "
    f"N={N_SAMPLES} RAAN samples, 1 orbital period each",
    fontsize=10,
)

ax = axes[0]
ax.hist(contacts / 60, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(
    contacts.mean() / 60,
    color="red",
    ls="--",
    lw=1.8,
    label=f"Mean (all): {contacts.mean()/60:.3f} min",
)
ax.axvline(
    ref_mean / 60,
    color="green",
    ls="--",
    lw=1.8,
    label=f"Mean (nz):  {ref_mean/60:.3f} min",
)
ax.axvline(
    np.median(contacts) / 60,
    color="orange",
    ls="--",
    lw=1.8,
    label=f"Median (all): {np.median(contacts)/60:.3f} min",
)
ax.set_xlabel("Contact Time per Orbital Period (min)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Contact Times")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(np.rad2deg(raan_samples), contacts / 60, s=6, alpha=0.4, color="darkorange")
ax.axhline(
    contacts.mean() / 60,
    color="red",
    ls="--",
    lw=1.5,
    label=f"Mean: {contacts.mean()/60:.3f} min",
)
ax.axhline(
    ref_mean / 60,
    color="green",
    ls="--",
    lw=1.5,
    label=f"NZ Mean: {ref_mean/60:.3f} min",
)
ax.set_xlabel("RAAN (deg)")
ax.set_ylabel("Contact Time (min)")
ax.set_title("Contact Time vs RAAN")
ax.set_xlim(0, 360)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "montecarlo_gps_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved: {out_path}")
