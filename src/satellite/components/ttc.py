import numpy as np

from .base import Component


class Antenna(Component):
    def __init__(
        self,
        min_elevation_deg=30.0,
        boresight_body=np.array([1.0, 0.0, 0.0]),
        gain_dbi=0.0,
        pointing_loss_db=0.0,
        name="ttc_antenna",
    ):
        super().__init__(name)
        self.min_elevation_deg = float(min_elevation_deg)
        self.boresight_body = self._unit_vector(boresight_body)
        self.gain_dbi = float(gain_dbi)
        self.pointing_loss_db = float(pointing_loss_db)

    def _unit_vector(self, value):
        vector = np.asarray(value, dtype=float)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            raise ValueError("boresight_body must be non-zero.")
        return vector / norm


class LoRaRadio(Component):
    def __init__(
        self,
        frequency_hz=433e6,
        tx_power_dbw=-10.0,
        bandwidth_hz=125e3,
        spreading_factor=12,
        coding_rate=1,
        receiver_sensitivity_dbm=-136.0,
        ebn0_threshold_db=7.1,
        useful_bitrate_mbps=293e-6,
        tx_system_loss_db=1.0,
        rx_system_loss_db=1.0,
        atmospheric_loss_db=1.0,
        polarization_loss_db=3.0,
        name="lora_radio",
    ):
        super().__init__(name)
        self.frequency_hz = float(frequency_hz)
        self.tx_power_dbw = float(tx_power_dbw)
        self.bandwidth_hz = float(bandwidth_hz)
        self.spreading_factor = int(spreading_factor)
        self.coding_rate = int(coding_rate)
        self.receiver_sensitivity_dbm = float(receiver_sensitivity_dbm)
        self.ebn0_threshold_db = float(ebn0_threshold_db)
        self.useful_bitrate_mbps = float(useful_bitrate_mbps)
        self.tx_system_loss_db = float(tx_system_loss_db)
        self.rx_system_loss_db = float(rx_system_loss_db)
        self.atmospheric_loss_db = float(atmospheric_loss_db)
        self.polarization_loss_db = float(polarization_loss_db)


class BeaconSchedule(Component):
    def __init__(
        self,
        tx_seconds=15.0,
        rx_seconds=45.0,
        payload_downlink_bytes=51,
        payload_uplink_bytes=51,
        guard_seconds=3.0,
        name="beacon_schedule",
    ):
        super().__init__(name)
        self.tx_seconds = float(tx_seconds)
        self.rx_seconds = float(rx_seconds)
        self.payload_downlink_bytes = int(payload_downlink_bytes)
        self.payload_uplink_bytes = int(payload_uplink_bytes)
        self.guard_seconds = float(guard_seconds)

    @property
    def cycle_seconds(self):
        return self.tx_seconds + self.rx_seconds
