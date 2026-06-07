from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

from src import Satellite
from src.analysis.ttc import TtcContactMonitor
from src.external.ground import GroundStation
from src.satellite.components.ttc import Antenna
from src.constants import EARTH_MU, EARTH_RADIUS
from src.simulation import CircularOrbit, PrescribedOrbitSimulation
from src.utils.initial_conditions import generate_inclined_circular_orbit

GROUND_LAT_DEG = 28.3802
GROUND_LON_DEG = 75.6092
ALT_KM = 450.0
INC_DEG = 60.0
EL_MIN_DEG = 30.0

N_SAMPLES = 1000
DT = 0.1
SEED = 42

SMA = EARTH_RADIUS + ALT_KM
N_MOT = np.sqrt(EARTH_MU / SMA**3)
T_ORB = 2.0 * np.pi / N_MOT


def contact_time_for_raan(raan_deg, ground_station, timespan_s=None, dt=None):
    if timespan_s is None:
        timespan_s = T_ORB
    if dt is None:
        dt = DT

    r, v = generate_inclined_circular_orbit(
        inclination_deg=INC_DEG,
        radius_km=SMA,
        raan_deg=raan_deg,
        arg_lat_deg=0.0,
    )
    sat = Satellite()
    sat.add_component(Antenna(min_elevation_deg=EL_MIN_DEG))
    contact = sat.add_monitor(
        TtcContactMonitor(
            ground_station=ground_station,
            sample_period_s=dt,
        )
    )
    sat.position = r.copy()
    sat.velocity = v.copy()
    sat.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    sat.omega = np.zeros(3)

    orbit = CircularOrbit.from_satellite(sat)
    PrescribedOrbitSimulation(sat, orbit).run(timespan_s, dt)
    return contact.total_contact_time_s


def run_monte_carlo():
    rng = np.random.default_rng(SEED)
    raan_samples_deg = rng.uniform(0.0, 360.0, N_SAMPLES)
    ground_station = GroundStation(GROUND_LAT_DEG, GROUND_LON_DEG)

    print(
        f"Running Monte Carlo: {N_SAMPLES} RAAN samples, "
        f"1 orbital period each ({T_ORB/60:.2f} min), dt = {DT} s ..."
    )

    contacts = np.array(
        [
            contact_time_for_raan(raan_deg, ground_station)
            for raan_deg in raan_samples_deg
        ]
    )

    print("Done.\n")
    return raan_samples_deg, contacts


def print_results(contacts):
    nonzero = contacts[contacts > 0]
    ref_mean = nonzero.mean() if len(nonzero) else 1.0
    contact_pct = (contacts > 0).mean() * 100

    sep = "=" * 60
    print(sep)
    print("  Monte Carlo TTC Visibility -- Overall Results")
    print(sep)
    print(f"  Ground station : {GROUND_LAT_DEG} N, {GROUND_LON_DEG} E")
    print(f"  Orbit          : {ALT_KM:.0f} km, {INC_DEG:.0f} deg incl., circular")
    print(f"  Min elevation  : {EL_MIN_DEG:.0f} deg")
    print(f"  Orbital period : {T_ORB/60:.2f} min")
    print(f"  Samples        : {N_SAMPLES} | RAAN ~ Uniform[0, 360 deg]")
    print(f"  Time step      : {DT} s")
    print(sep)

    print(f"\n  Contact statistics across all {N_SAMPLES} RAAN samples")
    print("  (each satellite simulated for one full orbital period)\n")

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
        f"\n  Passes with any contact : {contact_pct:.1f}% "
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


def plot_results(raan_samples_deg, contacts):
    nonzero = contacts[contacts > 0]
    ref_mean = nonzero.mean() if len(nonzero) else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"MC TTC Visibility -- {ALT_KM:.0f} km, {INC_DEG:.0f} deg incl., "
        f"El>={EL_MIN_DEG:.0f} deg | "
        f"GS: {GROUND_LAT_DEG}N {GROUND_LON_DEG}E | "
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
        label=f"Mean (nz): {ref_mean/60:.3f} min",
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
    ax.scatter(raan_samples_deg, contacts / 60, s=6, alpha=0.4, color="darkorange")
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
    out_path = PROJECT_ROOT / "montecarlo_gps_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Plot saved: {out_path}")


if __name__ == "__main__":
    raan_samples, contacts = run_monte_carlo()
    print_results(contacts)
    plot_results(raan_samples, contacts)
