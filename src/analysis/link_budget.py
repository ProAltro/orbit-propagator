import numpy as np

from ..constants import EARTH_RADIUS, SPEED_OF_LIGHT

BOLTZMANN = 1.380649e-23


def slant_range_km(elevation_deg, altitude_km=450.0, earth_radius_km=EARTH_RADIUS):
    el = np.radians(elevation_deg)
    return float(
        np.sqrt((earth_radius_km + altitude_km) ** 2 - (earth_radius_km * np.cos(el)) ** 2)
        - earth_radius_km * np.sin(el)
    )


def link_budget(
    elevation_deg,
    range_km=None,
    radio=None,
    tx_antenna=None,
    rx_gain_dbi=8.0,
    rx_lna_gain_db=15.0,
    tx_gain_dbi=None,
    altitude_km=450.0,
    sky_temp_cold_k=200.0,
    sky_temp_hot_k=1000.0,
    required_snr_db=-20.0,
    matlab_ebn0_convention=True,
):
    """LoRa-style link budget.

    The default Eb/N0 path preserves the MATLAB-equivalent convention from
    updates/ttc.py: useful bitrate is represented in Mbps and the C/N0 offset is
    explicitly compensated by the Eb/N0 conversion.
    """
    if range_km is None:
        range_km = slant_range_km(elevation_deg, altitude_km=altitude_km)

    frequency_hz = getattr(radio, "frequency_hz", 433e6)
    tx_power_dbw = getattr(radio, "tx_power_dbw", -10.0)
    bandwidth_hz = getattr(radio, "bandwidth_hz", 125e3)
    receiver_sensitivity_dbm = getattr(radio, "receiver_sensitivity_dbm", -136.0)
    ebn0_threshold_db = getattr(radio, "ebn0_threshold_db", 7.1)
    useful_bitrate_mbps = getattr(radio, "useful_bitrate_mbps", 293e-6)
    tx_system_loss_db = getattr(radio, "tx_system_loss_db", 1.0)
    rx_system_loss_db = getattr(radio, "rx_system_loss_db", 1.0)
    atmospheric_loss_db = getattr(radio, "atmospheric_loss_db", 1.0)
    polarization_loss_db = getattr(radio, "polarization_loss_db", 3.0)

    if tx_gain_dbi is None:
        tx_gain_dbi = getattr(tx_antenna, "gain_dbi", 0.0)
    pointing_loss_db = getattr(tx_antenna, "pointing_loss_db", 1.0)

    range_m = float(range_km) * 1000.0
    fspl_db = 20.0 * np.log10(4.0 * np.pi * range_m * frequency_hz / SPEED_OF_LIGHT)
    eirp_dbw = tx_power_dbw + tx_gain_dbi - tx_system_loss_db
    pr_dbw = (
        eirp_dbw
        - fspl_db
        - atmospheric_loss_db
        - polarization_loss_db
        - pointing_loss_db
        + rx_gain_dbi
        + rx_lna_gain_db
        - rx_system_loss_db
    )
    pr_dbm = pr_dbw + 30.0

    el_rad = np.radians(elevation_deg)
    t_ant_k = sky_temp_cold_k + (sky_temp_hot_k - sky_temp_cold_k) * (
        1.0 - np.sin(el_rad)
    )
    noise_power_dbm = receiver_sensitivity_dbm - required_snr_db
    noise_figure_db = noise_power_dbm + 174.0 - 10.0 * np.log10(bandwidth_hz)
    noise_factor = 10.0 ** (noise_figure_db / 10.0)
    receiver_temp_k = (noise_factor - 1.0) * 290.0
    system_temp_k = t_ant_k + receiver_temp_k
    spec_temp_k = noise_factor * 290.0
    sensitivity_actual_dbm = receiver_sensitivity_dbm + 10.0 * np.log10(
        system_temp_k / spec_temp_k
    )
    sensitivity_margin_db = pr_dbm - sensitivity_actual_dbm

    carrier_for_cno_dbw = pr_dbw - 30.0 if matlab_ebn0_convention else pr_dbw
    noise_density_dbw_hz = 10.0 * np.log10(BOLTZMANN * system_temp_k)
    cno_dbhz = carrier_for_cno_dbw - noise_density_dbw_hz
    if matlab_ebn0_convention:
        ebn0_db = cno_dbhz - 10.0 * np.log10(useful_bitrate_mbps) - 60.0
    else:
        ebn0_db = cno_dbhz - 10.0 * np.log10(useful_bitrate_mbps * 1e6)
    link_margin_db = ebn0_db - ebn0_threshold_db

    return {
        "elevation_deg": float(elevation_deg),
        "range_km": float(range_km),
        "fspl_db": float(fspl_db),
        "noise_temperature_k": float(system_temp_k),
        "antenna_temperature_k": float(t_ant_k),
        "received_power_dbm": float(pr_dbm),
        "sensitivity_actual_dbm": float(sensitivity_actual_dbm),
        "sensitivity_margin_db": float(sensitivity_margin_db),
        "cno_dbhz": float(cno_dbhz),
        "ebn0_db": float(ebn0_db),
        "link_margin_db": float(link_margin_db),
        "link_ok": bool(link_margin_db >= 0.0),
    }


def lora_time_on_air(
    payload_bytes,
    spreading_factor=12,
    bandwidth_hz=125e3,
    coding_rate=1,
    preamble_symbols=8,
    explicit_header=True,
    crc=True,
):
    sf = int(spreading_factor)
    bandwidth = float(bandwidth_hz)
    symbol_time = (2**sf) / bandwidth
    preamble_time = (preamble_symbols + 4.25) * symbol_time
    de = 2 if sf >= 11 else 0
    ih = 0 if explicit_header else 1
    crc_flag = 1 if crc else 0
    payload_symbols = 8 + max(
        int(
            np.ceil(
                (8 * int(payload_bytes) - 4 * sf + 28 + 16 * crc_flag - 20 * ih)
                / (4 * (sf - de))
            )
        )
        * (int(coding_rate) + 4),
        0,
    )
    return float(preamble_time + payload_symbols * symbol_time)


def beacon_optimizer(
    downlink_bytes=51,
    uplink_bytes=51,
    guard_s=3.0,
    pass_duration_s=320.0,
    radio=None,
):
    spreading_factor = getattr(radio, "spreading_factor", 12)
    bandwidth_hz = getattr(radio, "bandwidth_hz", 125e3)
    coding_rate = getattr(radio, "coding_rate", 1)
    toa_dl = lora_time_on_air(downlink_bytes, spreading_factor, bandwidth_hz, coding_rate)
    toa_ul = lora_time_on_air(uplink_bytes, spreading_factor, bandwidth_hz, coding_rate)
    x_s = toa_dl + 0.5
    y_s = toa_ul + guard_s
    cycle_s = x_s + y_s
    windows = int(pass_duration_s / cycle_s) if cycle_s > 0.0 else 0
    return {
        "x_s": round(x_s, 2),
        "y_s": round(y_s, 2),
        "cycle_s": round(cycle_s, 2),
        "windows_per_pass": windows,
        "max_ul_bytes": windows * int(uplink_bytes),
        "p_hit_pct": round(100.0 * y_s / cycle_s, 1) if cycle_s > 0.0 else 0.0,
        "toa_dl_s": round(toa_dl, 3),
        "toa_ul_s": round(toa_ul, 3),
    }
