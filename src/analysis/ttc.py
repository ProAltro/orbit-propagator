import numpy as np

from .link_budget import link_budget
from .monitor import AnalysisMonitor
from ..ground.geometry import doppler_shift_hz, elevation_azimuth_range
from ..ground import GroundStation


class TtcContactMonitor(AnalysisMonitor):
    def __init__(
        self,
        ground_station: GroundStation,
        min_elevation_deg: float | None = None,
        sample_period_s: float | None = None,
        antenna_component_name: str | None = "ttc_antenna",
        name: str = "ttc_contact",
    ):
        super().__init__(name)
        self.ground_station = ground_station
        self.min_elevation_deg = (
            None if min_elevation_deg is None else float(min_elevation_deg)
        )
        self.min_elevation_rad = (
            None if min_elevation_deg is None else np.radians(self.min_elevation_deg)
        )
        self.sample_period_s = sample_period_s
        self.antenna_component_name = antenna_component_name

        self.times = []
        self.elevations_rad = []
        self.in_contact = []

    def attach(self, sat):
        self._resolve_min_elevation(sat)

    @property
    def elevations_deg(self):
        return np.degrees(self.elevations_rad)

    @property
    def total_contact_time_s(self):
        if self.sample_period_s is not None:
            return float(np.sum(self.in_contact) * self.sample_period_s)

        if len(self.times) < 2:
            return 0.0

        times = np.asarray(self.times)
        in_contact = np.asarray(self.in_contact, dtype=float)
        dt = np.diff(times, prepend=times[0])
        return float(np.sum(in_contact * dt))

    @property
    def contact_fraction(self):
        return float(np.mean(self.in_contact)) if self.in_contact else 0.0

    def sample(self, sat):
        min_elevation_rad = self._resolve_min_elevation(sat)
        elevation = self.elevation_angle(sat.position, sat.time)
        self.times.append(float(sat.time))
        self.elevations_rad.append(float(elevation))
        self.in_contact.append(bool(elevation >= min_elevation_rad))

    def elevation_angle(self, position_eci_km, time_s):
        elevation, _, _ = elevation_azimuth_range(
            position_eci_km,
            self.ground_station,
            time_s,
        )
        return elevation

    def _resolve_min_elevation(self, sat):
        if self.min_elevation_rad is not None:
            return self.min_elevation_rad

        if self.antenna_component_name is None:
            raise ValueError("min_elevation_deg is required when no antenna is used.")

        antenna = sat.get_component(self.antenna_component_name)
        self.min_elevation_deg = antenna.min_elevation_deg
        self.min_elevation_rad = np.radians(self.min_elevation_deg)
        return self.min_elevation_rad


class TtcLinkBudgetMonitor(TtcContactMonitor):
    def __init__(
        self,
        ground_station: GroundStation,
        min_elevation_deg: float | None = None,
        sample_period_s: float | None = None,
        antenna_component_name: str | None = "ttc_antenna",
        radio_component_name: str | None = "lora_radio",
        rx_gain_dbi: float = 8.0,
        rx_lna_gain_db: float = 15.0,
        name: str = "ttc_link_budget",
    ):
        super().__init__(
            ground_station,
            min_elevation_deg=min_elevation_deg,
            sample_period_s=sample_period_s,
            antenna_component_name=antenna_component_name,
            name=name,
        )
        self.radio_component_name = radio_component_name
        self.rx_gain_dbi = float(rx_gain_dbi)
        self.rx_lna_gain_db = float(rx_lna_gain_db)
        self.azimuths_rad = []
        self.ranges_km = []
        self.link_margins_db = []
        self.ebn0_db = []
        self.doppler_hz = []
        self.link_open = []
        self._previous_time = None
        self._previous_range_km = None

    def sample(self, sat):
        min_elevation_rad = self._resolve_min_elevation(sat)
        elevation, azimuth, range_km = elevation_azimuth_range(
            sat.position,
            self.ground_station,
            sat.time,
        )

        radio = (
            sat.get_component(self.radio_component_name, default=None)
            if self.radio_component_name is not None
            else None
        )
        antenna = (
            sat.get_component(self.antenna_component_name, default=None)
            if self.antenna_component_name is not None
            else None
        )
        budget = link_budget(
            np.degrees(elevation),
            range_km=range_km,
            radio=radio,
            tx_antenna=antenna,
            rx_gain_dbi=self.rx_gain_dbi,
            rx_lna_gain_db=self.rx_lna_gain_db,
        )

        doppler = 0.0
        if self._previous_time is not None and sat.time > self._previous_time:
            range_rate_m_s = (
                (range_km - self._previous_range_km)
                * 1000.0
                / (sat.time - self._previous_time)
            )
            frequency_hz = getattr(radio, "frequency_hz", 433e6)
            doppler = doppler_shift_hz(range_rate_m_s, frequency_hz)

        self.times.append(float(sat.time))
        self.elevations_rad.append(float(elevation))
        self.azimuths_rad.append(float(azimuth))
        self.ranges_km.append(float(range_km))
        self.in_contact.append(bool(elevation >= min_elevation_rad))
        self.link_margins_db.append(budget["link_margin_db"])
        self.ebn0_db.append(budget["ebn0_db"])
        self.doppler_hz.append(float(doppler))
        self.link_open.append(bool(elevation >= min_elevation_rad and budget["link_ok"]))
        self._previous_time = float(sat.time)
        self._previous_range_km = float(range_km)

    @property
    def azimuths_deg(self):
        return np.degrees(self.azimuths_rad)


def summarize_passes(times, elevations_deg, azimuths_deg=None, link_margins_db=None, doppler_hz=None, min_elevation_deg=5.0):
    summaries = []
    current = []
    for i, (time_s, elevation_deg) in enumerate(zip(times, elevations_deg)):
        if elevation_deg >= min_elevation_deg:
            current.append(i)
            continue
        if current:
            summaries.append(_summarize_pass(current, times, elevations_deg, azimuths_deg, link_margins_db, doppler_hz))
            current = []
    if current:
        summaries.append(_summarize_pass(current, times, elevations_deg, azimuths_deg, link_margins_db, doppler_hz))
    return summaries


def _summarize_pass(indices, times, elevations_deg, azimuths_deg, link_margins_db, doppler_hz):
    peak_index = max(indices, key=lambda i: elevations_deg[i])
    start = indices[0]
    end = indices[-1]
    return {
        "start_s": float(times[start]),
        "end_s": float(times[end]),
        "duration_s": float(times[end] - times[start]),
        "peak_time_s": float(times[peak_index]),
        "peak_elevation_deg": float(elevations_deg[peak_index]),
        "aos_azimuth_deg": None if azimuths_deg is None else float(azimuths_deg[start]),
        "los_azimuth_deg": None if azimuths_deg is None else float(azimuths_deg[end]),
        "best_link_margin_db": None
        if link_margins_db is None
        else float(max(link_margins_db[i] for i in indices)),
        "doppler_at_peak_hz": None
        if doppler_hz is None
        else float(doppler_hz[peak_index]),
    }
