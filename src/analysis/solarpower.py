import numpy as np

from .monitor import AnalysisMonitor
from ..maths.maths import A_from_q


class SolarPowerMonitor(AnalysisMonitor):
    def __init__(
        self,
        name="solar_power",
        sun_position_km=np.array([1.496e8, 0.0, 0.0]),
        solar_constant=1361.0,
        panel_efficiency=None,
        earth_radius_km=6371.0,
        solar_panel_areas=None,
        solar_panel_component_name="solar_panels",
    ):
        super().__init__(name)
        self.sun_position_km = np.asarray(sun_position_km, dtype=float)
        self.solar_constant = float(solar_constant)
        self.panel_efficiency = None if panel_efficiency is None else float(panel_efficiency)
        self.earth_radius_km = float(earth_radius_km)
        self.solar_panel_areas = (
            None
            if solar_panel_areas is None
            else np.asarray(solar_panel_areas, dtype=float)
        )
        self.solar_panel_component_name = solar_panel_component_name

        if np.linalg.norm(self.sun_position_km) == 0.0:
            raise ValueError("sun_position_km must be non-zero.")

        self.times = []
        self.powers = []
        self.in_eclipse = []
        self.sun_dirs_body = []

    def attach(self, sat):
        self._solar_panel_component(sat)

    @property
    def mean_power(self):
        return float(np.mean(self.powers)) if self.powers else 0.0

    @property
    def max_power(self):
        return float(np.max(self.powers)) if self.powers else 0.0

    @property
    def min_power(self):
        return float(np.min(self.powers)) if self.powers else 0.0

    @property
    def eclipse_fraction(self):
        return float(np.mean(self.in_eclipse)) if self.in_eclipse else 0.0

    def sample(self, sat):
        r_sat_sun = self.sun_position_km - sat.position
        r_sat_sun_norm = np.linalg.norm(r_sat_sun)
        if r_sat_sun_norm == 0.0:
            raise ValueError("Satellite position cannot equal sun_position_km.")

        sun_dir_eci = r_sat_sun / r_sat_sun_norm
        sun_dir_body = A_from_q(sat.quaternion) @ sun_dir_eci
        eclipsed = self._is_eclipsed(sat.position)

        if eclipsed:
            power = 0.0
        else:
            solar_panels = self._solar_panel_component(sat)
            power = solar_panels.power_from_sun_body(
                sat,
                sun_dir_body,
                self.solar_constant,
                areas_m2=self.solar_panel_areas,
                efficiency=self.panel_efficiency,
            )

        self.times.append(float(sat.time))
        self.powers.append(float(power))
        self.in_eclipse.append(bool(eclipsed))
        self.sun_dirs_body.append(sun_dir_body.copy())

    def _is_eclipsed(self, position_km):
        position_norm = np.linalg.norm(position_km)
        if position_norm == 0.0:
            return False

        sun_dir = self.sun_position_km / np.linalg.norm(self.sun_position_km)
        projected_distance = np.dot(position_km, sun_dir)
        perpendicular_sq = position_norm**2 - projected_distance**2
        perpendicular_distance = np.sqrt(max(perpendicular_sq, 0.0))

        return projected_distance < 0.0 and perpendicular_distance < self.earth_radius_km

    def _solar_panel_component(self, sat):
        solar_panels = sat.get_component(self.solar_panel_component_name)

        areas = (
            solar_panels.areas_m2
            if self.solar_panel_areas is None
            else self.solar_panel_areas
        )
        expected_shape = solar_panels.areas_m2.shape
        if areas.shape != expected_shape:
            raise ValueError("solar panel areas must match satellite face normals.")

        return solar_panels
