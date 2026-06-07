import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import Satellite
from src.analysis.detumble import AngularRateMonitor
from src.analysis.dynamics import DynamicsEnvironmentMonitor
from src.satellite.components import Magnetorquer
from src.simulation import AngularRateBelowStop, DynamicSimulation
from src.utils.initial_conditions import keplerian_to_state_vectors


def parse_vector_deg_s(value):
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values.")
    return np.radians(np.array(parts))


def format_duration(seconds):
    hours = seconds / 3600.0
    if hours < 48.0:
        return f"{hours:.2f} h"
    return f"{hours / 24.0:.2f} days"


def build_satellite(args):
    sat = Satellite()
    sat.position, sat.velocity = keplerian_to_state_vectors(
        altitude_km=args.altitude_km,
        inclination_deg=args.inclination_deg,
        raan_deg=args.raan_deg,
        argument_of_perigee_deg=args.argument_of_perigee_deg,
        true_anomaly_deg=args.true_anomaly_deg,
    )
    sat.mass = args.mass_kg
    sat.J = np.diag(args.inertia_kg_m2)
    sat.J_inv = np.linalg.inv(sat.J)
    sat.omega = args.initial_omega_rad_s
    sat.add_component(
        Magnetorquer(
            max_dipole_a_m2=args.max_dipole_a_m2,
            bdot_gain=args.bdot_gain,
        )
    )
    return sat


def run_detumble(args):
    sat = build_satellite(args)
    threshold_rad_s = np.radians(args.threshold_deg_s)
    rate_monitor = sat.add_monitor(AngularRateMonitor(threshold_rad_s))
    env_monitor = sat.add_monitor(DynamicsEnvironmentMonitor())

    initial_rate = float(np.linalg.norm(sat.omega))
    max_time_s = args.max_hours * 3600.0
    DynamicSimulation(sat).run(
        total_time_s=max_time_s,
        dt_s=args.dt,
        stop_condition=AngularRateBelowStop(threshold_rad_s),
    )

    return sat, rate_monitor, env_monitor, initial_rate, threshold_rad_s


def print_report(args, sat, rate_monitor, env_monitor, initial_rate, threshold_rad_s):
    final_rate = rate_monitor.rate_norm_rad_s[-1]
    detumble_time = rate_monitor.detumble_time_s
    torques = np.asarray(env_monitor.magnetorquer_torques_body_n_m)
    dipoles = [
        component.last_dipole_a_m2
        for component in sat.get_components(Magnetorquer)
    ]

    print("Magnetorquer detumble estimate")
    print("------------------------------")
    print(
        f"Orbit: altitude={args.altitude_km:.0f} km, inc={args.inclination_deg:.1f} deg, "
        f"RAAN={args.raan_deg:.1f} deg"
    )
    print(
        f"Initial omega: {np.degrees(args.initial_omega_rad_s)} deg/s "
        f"(|omega|={np.degrees(initial_rate):.3f} deg/s)"
    )
    print(f"Threshold: {args.threshold_deg_s:.3f} deg/s")
    print(
        f"Magnetorquer: max dipole={args.max_dipole_a_m2:g} A m^2, "
        f"B-dot gain={args.bdot_gain:g}"
    )
    print(f"Inertia diag: {args.inertia_kg_m2} kg m^2")
    print(f"Step: {args.dt:g} s, simulated: {format_duration(rate_monitor.times[-1])}")

    if detumble_time is None:
        print(
            f"Result: not detumbled within {format_duration(args.max_hours * 3600.0)}. "
            f"Final |omega|={np.degrees(final_rate):.3f} deg/s"
        )
    else:
        print(
            f"Result: detumbled in about {format_duration(detumble_time)} "
            f"({detumble_time:.0f} s)."
        )

    if len(torques):
        torque_norms = np.linalg.norm(torques, axis=1)
        print(
            f"Torque norm: mean={np.mean(torque_norms):.3e} N m, "
            f"max={np.max(torque_norms):.3e} N m"
        )
    if dipoles:
        print(f"Last commanded dipole: {dipoles[0]} A m^2")

    print()
    print("Caveat: this is a first-order B-dot/titled-dipole estimate. It ignores sensor")
    print("noise, coil electrical limits, duty cycling, residual dipoles, and IGRF fidelity.")


def main():
    parser = argparse.ArgumentParser(
        description="Rough B-dot magnetorquer detumble-time estimate."
    )
    parser.add_argument("--altitude-km", type=float, default=450.0)
    parser.add_argument("--inclination-deg", type=float, default=51.6)
    parser.add_argument("--raan-deg", type=float, default=60.0)
    parser.add_argument("--argument-of-perigee-deg", type=float, default=0.0)
    parser.add_argument("--true-anomaly-deg", type=float, default=0.0)
    parser.add_argument(
        "--initial-omega-deg-s",
        dest="initial_omega_rad_s",
        type=parse_vector_deg_s,
        default=np.radians(np.array([5.0, -3.0, 2.0])),
        help='Initial body rate as "wx,wy,wz" in deg/s.',
    )
    parser.add_argument("--threshold-deg-s", type=float, default=0.1)
    parser.add_argument("--max-dipole-a-m2", type=float, default=0.02)
    parser.add_argument("--bdot-gain", type=float, default=1e6)
    parser.add_argument("--mass-kg", type=float, default=1.0)
    parser.add_argument(
        "--inertia-kg-m2",
        type=float,
        nargs=3,
        default=[0.0027, 0.0027, 0.0054],
        help="Principal inertia diagonal in kg m^2.",
    )
    parser.add_argument("--dt", type=float, default=10.0)
    parser.add_argument("--max-hours", type=float, default=24.0)
    args = parser.parse_args()

    sat, rate_monitor, env_monitor, initial_rate, threshold_rad_s = run_detumble(args)
    print_report(args, sat, rate_monitor, env_monitor, initial_rate, threshold_rad_s)


if __name__ == "__main__":
    main()
