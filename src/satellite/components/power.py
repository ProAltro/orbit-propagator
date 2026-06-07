import numpy as np

from .base import Component


DEFAULT_SOLAR_PANEL_AREAS_M2 = np.array([0.006, 0.006, 0.006, 0.006, 0.006, 0.006])


class SolarPanelArray(Component):
    def __init__(
        self,
        areas_m2=DEFAULT_SOLAR_PANEL_AREAS_M2,
        efficiency=0.2,
        surface_model_name="body_surfaces",
        name="solar_panels",
    ):
        super().__init__(name)
        self.areas_m2 = np.asarray(areas_m2, dtype=float)
        self.efficiency = float(efficiency)
        self.surface_model_name = surface_model_name

    def attach(self, sat):
        surface_model = sat.get_component(self.surface_model_name)
        if self.areas_m2.shape != (len(surface_model.normals_body),):
            raise ValueError("solar panel areas must match surface normal length.")

    def power_from_sun_body(
        self,
        sat,
        sun_dir_body,
        solar_constant,
        areas_m2=None,
        efficiency=None,
    ):
        surface_model = sat.get_component(self.surface_model_name)
        areas = self.areas_m2 if areas_m2 is None else np.asarray(areas_m2, dtype=float)
        if areas.shape != self.areas_m2.shape:
            raise ValueError("solar panel areas must match configured panel areas.")
        panel_efficiency = self.efficiency if efficiency is None else float(efficiency)
        cos_angles = surface_model.normals_body @ sun_dir_body
        illuminated = np.maximum(cos_angles, 0.0)
        return float(solar_constant * panel_efficiency * np.sum(areas * illuminated))
