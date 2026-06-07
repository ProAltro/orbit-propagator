import unittest

import numpy as np

from src import Satellite
from src.analysis.attitude import BodyAxisAlignmentMonitor
from src.analysis.link_budget import beacon_optimizer, link_budget, lora_time_on_air
from src.analysis.monitor import AnalysisMonitor
from src.analysis.solarpower import SolarPowerMonitor
from src.analysis.ttc import TtcContactMonitor, TtcLinkBudgetMonitor
from src.dynamics.magnetic_torque import magnetorquer_torque_body
from src.dynamics.providers import (
    DefaultDynamicsModel,
    ForceProvider,
    MagnetorquerTorqueProvider,
)
from src.dynamics.solar_radiation_force import solar_radiation_acceleration
from src.dynamics.state import DynamicsState
from src.environment.magnetic import earth_magnetic_field_eci
from src.environment.sun import sun_direction_unit
from src.ground import GroundStation
from src.maths.maths import ecef_to_eci, eci_to_ecef
from src.satellite.components.actuators import Magnetorquer
from src.satellite.components.power import SolarPanelArray
from src.satellite.components.surfaces import BodySurfaceModel
from src.satellite.components.ttc import Antenna, LoRaRadio
from src.simulation import (
    AngularRateBelowStop,
    DynamicSimulation,
    PrescribedOrbitSimulation,
)
from src.simulation.orbit import CircularOrbit, simulate_circular_orbit
from src.utils.initial_conditions import (
    generate_inclined_circular_orbit,
    keplerian_to_state_vectors,
)


class DummyMonitor(AnalysisMonitor):
    def __init__(self):
        super().__init__("dummy")
        self.attached_sat = None
        self.sample_times = []

    def attach(self, sat):
        self.attached_sat = sat

    def sample(self, sat):
        self.sample_times.append(sat.time)


def satellite_at(position):
    sat = Satellite()
    sat.position = np.asarray(position, dtype=float)
    sat.velocity = np.array([0.0, 7.67, 0.0])
    sat.quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    sat.omega = np.zeros(3)
    return sat


class ZeroForceProvider:
    def acceleration_eci_km_s2(self, time_s, state, sat):
        return np.zeros(3)


class AnalysisMonitorTests(unittest.TestCase):
    def test_satellite_has_default_hardware_components(self):
        sat = Satellite()

        self.assertIsInstance(sat.get_component("body_surfaces"), BodySurfaceModel)
        self.assertIsInstance(sat.get_component("solar_panels"), SolarPanelArray)
        self.assertEqual(sat.n.shape, (6, 3))
        self.assertEqual(sat.A.shape, (6,))
        self.assertEqual(sat.solarp_area.shape, (6,))
        self.assertEqual(len(sat.get_components(SolarPanelArray)), 1)

    def test_satellite_can_be_created_without_default_components(self):
        sat = Satellite(install_default_components=False)

        self.assertEqual(sat.get_components(), [])
        with self.assertRaises(KeyError):
            sat.get_component("body_surfaces")

    def test_satellite_rejects_duplicate_component_names(self):
        sat = Satellite()

        with self.assertRaises(ValueError):
            sat.add_component(BodySurfaceModel())

    def test_satellite_samples_attached_monitor_after_propagate(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        monitor = sat.add_monitor(DummyMonitor())

        sat.propagate(1.0)

        self.assertIs(monitor.attached_sat, sat)
        self.assertEqual(len(monitor.sample_times), 1)
        self.assertEqual(monitor.sample_times[0], sat.time)

    def test_solar_power_is_positive_when_panel_faces_sun(self):
        sat = satellite_at([0.0, 6778.0, 0.0])
        monitor = SolarPowerMonitor(
            sun_position_km=np.array([1.496e8, 0.0, 0.0]),
            solar_panel_areas=np.ones(6),
        )

        monitor.sample(sat)

        self.assertFalse(monitor.in_eclipse[-1])
        self.assertGreater(monitor.powers[-1], 0.0)

    def test_solar_power_monitor_uses_solar_panel_component(self):
        sat = satellite_at([0.0, 6778.0, 0.0])
        sat.get_component("solar_panels").areas_m2 = np.ones(6)
        monitor = SolarPowerMonitor(sun_position_km=np.array([1.496e8, 0.0, 0.0]))

        monitor.sample(sat)

        self.assertGreater(monitor.powers[-1], 0.0)

    def test_solar_power_monitor_does_not_mutate_panel_component_overrides(self):
        sat = satellite_at([0.0, 6778.0, 0.0])
        panels = sat.get_component("solar_panels")
        original_areas = panels.areas_m2.copy()
        original_efficiency = panels.efficiency
        monitor = SolarPowerMonitor(
            sun_position_km=np.array([1.496e8, 0.0, 0.0]),
            solar_panel_areas=np.ones(6),
            panel_efficiency=0.5,
        )

        monitor.sample(sat)

        np.testing.assert_allclose(panels.areas_m2, original_areas)
        self.assertEqual(panels.efficiency, original_efficiency)

    def test_solar_power_is_zero_in_eclipse(self):
        sat = satellite_at([-6778.0, 0.0, 0.0])
        monitor = SolarPowerMonitor(
            sun_position_km=np.array([1.496e8, 0.0, 0.0]),
            solar_panel_areas=np.ones(6),
        )

        monitor.sample(sat)

        self.assertTrue(monitor.in_eclipse[-1])
        self.assertEqual(monitor.powers[-1], 0.0)

    def test_solar_monitor_histories_have_matching_lengths(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        monitor = sat.add_monitor(SolarPowerMonitor())

        for _ in range(3):
            sat.propagate(1.0)

        self.assertEqual(len(monitor.times), 3)
        self.assertEqual(len(monitor.powers), 3)
        self.assertEqual(len(monitor.in_eclipse), 3)
        self.assertEqual(len(monitor.sun_dirs_body), 3)

    def test_circular_orbit_simulation_samples_arbitrary_monitors(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        dummy = sat.add_monitor(DummyMonitor())
        alignment = sat.add_monitor(BodyAxisAlignmentMonitor())

        simulate_circular_orbit(
            sat,
            total_time=30.0,
            dt=10.0,
        )

        self.assertEqual(len(dummy.sample_times), 3)
        self.assertEqual(len(alignment.alignments), 3)

    def test_circular_orbit_class_samples_arbitrary_monitors(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        dummy = sat.add_monitor(DummyMonitor())
        orbit = CircularOrbit.from_satellite(sat)

        orbit.simulate(sat, total_time=20.0, dt=10.0)

        self.assertEqual(len(dummy.sample_times), 2)

    def test_dynamic_simulation_samples_monitors_and_advances_time(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        dummy = sat.add_monitor(DummyMonitor())

        DynamicSimulation(sat).run(total_time_s=2.0, dt_s=1.0)

        self.assertEqual(sat.time, 2.0)
        self.assertEqual(len(dummy.sample_times), 2)

    def test_prescribed_orbit_simulation_updates_position_and_velocity(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        dummy = sat.add_monitor(DummyMonitor())
        orbit = CircularOrbit.from_satellite(sat)

        PrescribedOrbitSimulation(sat, orbit).run(total_time_s=20.0, dt_s=10.0)

        np.testing.assert_allclose(sat.position, orbit.position_at(20.0))
        np.testing.assert_allclose(sat.velocity, orbit.velocity_at(20.0))
        self.assertEqual(len(dummy.sample_times), 2)

    def test_angular_rate_stop_condition_stops_dynamic_simulation(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        sat.omega = np.zeros(3)
        dummy = sat.add_monitor(DummyMonitor())

        DynamicSimulation(sat).run(
            total_time_s=10.0,
            dt_s=1.0,
            stop_condition=AngularRateBelowStop(0.1),
        )

        self.assertEqual(sat.time, 0.0)
        self.assertEqual(len(dummy.sample_times), 0)

    def test_circular_orbit_uses_satellite_attitude_state(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        sat.omega = np.array([0.0, 0.0, 0.1])
        orbit = CircularOrbit.from_satellite(sat)

        orbit.simulate(sat, total_time=20.0, dt=10.0)

        self.assertFalse(np.allclose(sat.quaternion, np.array([1.0, 0.0, 0.0, 0.0])))

    def test_circular_orbit_velocity_at_is_circular_and_perpendicular(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        orbit = CircularOrbit.from_satellite(sat)

        position = orbit.position_at(123.0)
        velocity = orbit.velocity_at(123.0)

        self.assertAlmostEqual(np.dot(position, velocity), 0.0, places=8)
        self.assertAlmostEqual(np.linalg.norm(velocity), orbit.speed_km_s)

    def test_generate_inclined_circular_orbit_returns_perpendicular_state(self):
        r, v = generate_inclined_circular_orbit(
            inclination_deg=60.0,
            radius_km=6771.0,
            raan_deg=10.0,
            arg_lat_deg=25.0,
        )

        self.assertAlmostEqual(np.linalg.norm(r), 6771.0)
        self.assertAlmostEqual(np.dot(r, v), 0.0, places=8)

    def test_ttc_contact_monitor_detects_overhead_contact(self):
        sat = satellite_at([6878.0, 0.0, 0.0])
        ground_station = GroundStation(latitude_deg=0.0, longitude_deg=0.0)
        monitor = TtcContactMonitor(
            ground_station=ground_station,
            min_elevation_deg=30.0,
            sample_period_s=1.0,
        )

        monitor.sample(sat)

        self.assertTrue(monitor.in_contact[-1])
        self.assertGreater(monitor.total_contact_time_s, 0.0)

    def test_eci_to_ecef_is_identity_at_zero_time(self):
        position = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(eci_to_ecef(position, 0.0), position)

    def test_ecef_eci_round_trip(self):
        position = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(
            eci_to_ecef(ecef_to_eci(position, 1234.0), 1234.0),
            position,
            atol=1e-12,
        )

    def test_keplerian_state_generation_preserves_circular_geometry(self):
        r, v = keplerian_to_state_vectors(
            altitude_km=400.0,
            inclination_deg=51.6,
            raan_deg=60.0,
            argument_of_perigee_deg=0.0,
            true_anomaly_deg=20.0,
        )

        self.assertAlmostEqual(np.linalg.norm(r), 6778.137)
        self.assertAlmostEqual(np.dot(r, v), 0.0, places=8)

    def test_sun_direction_is_unit_length(self):
        direction = sun_direction_unit(86400.0)

        self.assertAlmostEqual(np.linalg.norm(direction), 1.0)

    def test_magnetic_field_is_finite_and_nonzero_in_leo(self):
        field = earth_magnetic_field_eci(np.array([6778.0, 0.0, 0.0]), 0.0)

        self.assertTrue(np.all(np.isfinite(field)))
        self.assertGreater(np.linalg.norm(field), 0.0)

    def test_srp_zero_in_eclipse(self):
        sat = satellite_at(-6778.0 * sun_direction_unit(0.0))

        acceleration = solar_radiation_acceleration(0.0, sat.position, sat)

        np.testing.assert_allclose(acceleration, np.zeros(3))

    def test_srp_nonzero_when_illuminated(self):
        sun_dir = sun_direction_unit(0.0)
        perpendicular = np.cross(sun_dir, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(perpendicular) == 0.0:
            perpendicular = np.array([1.0, 0.0, 0.0])
        sat = satellite_at(6778.0 * perpendicular / np.linalg.norm(perpendicular))

        acceleration = solar_radiation_acceleration(0.0, sat.position, sat)

        self.assertGreater(np.linalg.norm(acceleration), 0.0)

    def test_magnetorquer_initial_history_returns_zero_torque(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        sat.add_component(Magnetorquer())

        torque = magnetorquer_torque_body(sat.position, sat.time, sat.quaternion, sat, 1.0)

        np.testing.assert_allclose(torque, np.zeros(3))

    def test_default_torque_provider_returns_zero_without_magnetorquer(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        provider = MagnetorquerTorqueProvider()

        torque = provider.torque_body_n_m(
            sat.time,
            DynamicsState.from_satellite(sat),
            sat,
            1.0,
        )

        np.testing.assert_allclose(torque, np.zeros(3))

    def test_default_dynamics_model_sums_zero_force_provider_stably(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        model = DefaultDynamicsModel(force_providers=[ZeroForceProvider()])

        acceleration = model.acceleration_eci_km_s2(
            sat.time,
            DynamicsState.from_satellite(sat),
            sat,
        )

        np.testing.assert_allclose(acceleration, np.zeros(3))

    def test_magnetorquer_torque_is_perpendicular_to_dipole_and_field(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        magnetorquer = sat.add_component(Magnetorquer(max_dipole_a_m2=1.0))

        magnetorquer_torque_body(sat.position, 0.0, sat.quaternion, sat, 1.0)
        torque = magnetorquer_torque_body(
            np.array([0.0, 6778.0, 0.0]),
            10.0,
            sat.quaternion,
            sat,
            10.0,
        )
        field = earth_magnetic_field_eci(np.array([0.0, 6778.0, 0.0]), 10.0)

        self.assertAlmostEqual(np.dot(torque, magnetorquer.last_dipole_a_m2), 0.0, places=15)
        self.assertAlmostEqual(np.dot(torque, field), 0.0, places=15)

    def test_attitude_propagation_changes_omega_with_magnetorquer(self):
        sat = satellite_at([6778.0, 0.0, 0.0])
        sat.add_component(Magnetorquer(max_dipole_a_m2=100.0, bdot_gain=1e12))

        sat.propagate(10.0)
        before = sat.omega.copy()
        sat.propagate(10.0)

        self.assertFalse(np.allclose(sat.omega, before))

    def test_link_budget_known_default_closes_above_high_elevation(self):
        low = link_budget(5.0)
        high = link_budget(90.0)

        self.assertFalse(low["link_ok"])
        self.assertTrue(high["link_ok"])

    def test_lora_toa_and_beacon_optimizer_are_positive(self):
        self.assertGreater(lora_time_on_air(51), 0.0)
        self.assertGreater(beacon_optimizer(pass_duration_s=320.0)["windows_per_pass"], 0)

    def test_ttc_link_budget_monitor_records_link_state(self):
        sat = satellite_at([6878.0, 0.0, 0.0])
        sat.add_component(Antenna(min_elevation_deg=5.0, pointing_loss_db=1.0))
        sat.add_component(LoRaRadio())
        ground_station = GroundStation(latitude_deg=0.0, longitude_deg=0.0)
        monitor = TtcLinkBudgetMonitor(ground_station=ground_station, min_elevation_deg=5.0)

        monitor.sample(sat)

        self.assertEqual(len(monitor.link_margins_db), 1)
        self.assertTrue(monitor.link_open[-1])

    def test_ttc_contact_monitor_rejects_below_horizon_contact(self):
        sat = satellite_at([-6878.0, 0.0, 0.0])
        ground_station = GroundStation(latitude_deg=0.0, longitude_deg=0.0)
        monitor = TtcContactMonitor(
            ground_station=ground_station,
            min_elevation_deg=30.0,
            sample_period_s=1.0,
        )

        monitor.sample(sat)

        self.assertFalse(monitor.in_contact[-1])
        self.assertEqual(monitor.total_contact_time_s, 0.0)

    def test_ttc_contact_monitor_can_use_antenna_component_threshold(self):
        sat = satellite_at([6878.0, 0.0, 0.0])
        sat.add_component(Antenna(min_elevation_deg=30.0))
        ground_station = GroundStation(latitude_deg=0.0, longitude_deg=0.0)
        monitor = sat.add_monitor(
            TtcContactMonitor(ground_station=ground_station, sample_period_s=1.0)
        )

        monitor.sample(sat)

        self.assertEqual(monitor.min_elevation_deg, 30.0)
        self.assertTrue(monitor.in_contact[-1])


if __name__ == "__main__":
    unittest.main()
