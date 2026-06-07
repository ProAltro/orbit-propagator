import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.link_budget import beacon_optimizer, link_budget, lora_time_on_air
from src.analysis.ttc import TtcLinkBudgetMonitor, summarize_passes
from src.external.ground import GroundStation
from src import Satellite
from src.satellite.components import Antenna, LoRaRadio
from src.simulation import KeplerianCircularOrbit, PrescribedOrbitSimulation


PILANI = GroundStation(28.3802, 75.6092, altitude_km=0.333)

INITIAL_CONDITIONS = [
    ("IC-1 ISS-inclined RAAN=60", 450.0, 51.6, 60.0, 0.0, 0.0),
    ("IC-2 sun-synchronous RAAN=120", 450.0, 97.4, 120.0, 0.0, 0.0),
    ("IC-3 ISS-inclined RAAN=150", 450.0, 51.6, 150.0, 0.0, 0.0),
    ("IC-4 near-polar RAAN=0", 450.0, 85.0, 0.0, 0.0, 0.0),
    ("IC-5 low-inclination RAAN=75", 450.0, 28.5, 75.0, 0.0, 0.0),
]


def print_link_budget_table():
    print("Link budget: RFM96W/Lora @ 433 MHz, SF12, BW=125 kHz")
    print(f"{'El':>5} {'Range km':>9} {'FSPL':>8} {'C/N0':>9} {'Eb/N0':>9} {'LM':>8} Status")
    for elevation in range(5, 91, 5):
        budget = link_budget(elevation)
        status = "OPEN" if budget["link_ok"] else "CLOSED"
        print(
            f"{elevation:5.0f} {budget['range_km']:9.1f} {budget['fspl_db']:8.2f} "
            f"{budget['cno_dbhz']:9.2f} {budget['ebn0_db']:9.2f} "
            f"{budget['link_margin_db']:8.2f} {status}"
        )


def print_toa_table():
    print("\nLoRa time-on-air: SF12, BW=125 kHz, CR=4/5")
    print(f"{'Payload B':>9} {'ToA s':>8} {'Net bps':>8}")
    for payload in [10, 20, 30, 51, 64, 100, 128, 200, 255]:
        toa = lora_time_on_air(payload)
        print(f"{payload:9d} {toa:8.3f} {payload * 8 / toa:8.1f}")


def build_satellite():
    sat = Satellite()
    sat.add_component(Antenna(min_elevation_deg=5.0, gain_dbi=0.0, pointing_loss_db=1.0))
    sat.add_component(LoRaRadio())
    return sat


def run_ic(name, altitude_km, inclination_deg, raan_deg, aop_deg, ta_deg, days, dt):
    sat = build_satellite()
    orbit = KeplerianCircularOrbit(
        altitude_km,
        inclination_deg,
        raan_deg=raan_deg,
        argument_of_perigee_deg=aop_deg,
        true_anomaly_deg=ta_deg,
    )
    monitor = sat.add_monitor(
        TtcLinkBudgetMonitor(PILANI, min_elevation_deg=5.0, sample_period_s=dt)
    )
    PrescribedOrbitSimulation(sat, orbit).run(total_time_s=days * 86400.0, dt_s=dt)
    summaries = summarize_passes(
        monitor.times,
        monitor.elevations_deg,
        azimuths_deg=monitor.azimuths_deg,
        link_margins_db=monitor.link_margins_db,
        doppler_hz=monitor.doppler_hz,
        min_elevation_deg=5.0,
    )
    return name, orbit, summaries


def print_pass_report(name, orbit, summaries):
    print(f"\n{name}")
    print(
        f"Altitude={orbit.altitude_km:.0f} km, inc={orbit.inclination_deg:.1f} deg, "
        f"RAAN={orbit.raan_deg:.1f} deg, period={2*np.pi/orbit.mean_motion_rad_s/60:.2f} min"
    )
    if not summaries:
        print("No passes above 5 deg.")
        return

    good = sum(1 for item in summaries if item["peak_elevation_deg"] >= 45.0)
    print(f"Passes={len(summaries)}, good passes >=45 deg={good}")
    print(f"{'#':>3} {'Start h':>8} {'Dur s':>7} {'Peak el':>8} {'AOS az':>7} {'LOS az':>7} {'Best LM':>8} {'Dopp kHz':>9}")
    for index, item in enumerate(summaries[:20], start=1):
        print(
            f"{index:3d} {item['start_s']/3600:8.2f} {item['duration_s']:7.0f} "
            f"{item['peak_elevation_deg']:8.1f} {item['aos_azimuth_deg']:7.1f} "
            f"{item['los_azimuth_deg']:7.1f} {item['best_link_margin_db']:8.2f} "
            f"{item['doppler_at_peak_hz']/1000:9.2f}"
        )
    if len(summaries) > 20:
        print(f"... {len(summaries) - 20} additional passes omitted")

    durations = [item["duration_s"] for item in summaries if item["peak_elevation_deg"] >= 45.0]
    representative_duration = float(np.mean(durations)) if durations else float(np.mean([item["duration_s"] for item in summaries]))
    print(f"Beacon recommendation for representative pass {representative_duration:.0f} s:")
    print(beacon_optimizer(pass_duration_s=representative_duration))


def main():
    parser = argparse.ArgumentParser(description="Structured TTC link/pass report")
    parser.add_argument("--days", type=float, default=7.0, help="Simulation duration in days")
    parser.add_argument("--dt", type=float, default=10.0, help="Sample step in seconds")
    parser.add_argument("--limit-ics", type=int, default=None, help="Only run first N initial conditions")
    args = parser.parse_args()

    print_link_budget_table()
    print_toa_table()

    results = []
    for ic in INITIAL_CONDITIONS[: args.limit_ics]:
        result = run_ic(*ic, days=args.days, dt=args.dt)
        results.append(result)
        print_pass_report(*result)

    print("\nCross-IC summary")
    print(f"{'IC':>4} {'Passes':>7} {'Good':>5} {'Best el':>8} {'Best LM':>8}")
    for index, (name, _, summaries) in enumerate(results, start=1):
        if summaries:
            best = max(summaries, key=lambda item: item["peak_elevation_deg"])
            good = sum(1 for item in summaries if item["peak_elevation_deg"] >= 45.0)
            print(
                f"{index:4d} {len(summaries):7d} {good:5d} "
                f"{best['peak_elevation_deg']:8.1f} {best['best_link_margin_db']:8.2f}  {name}"
            )
        else:
            print(f"{index:4d} {0:7d} {0:5d} {'--':>8} {'--':>8}  {name}")


if __name__ == "__main__":
    main()
