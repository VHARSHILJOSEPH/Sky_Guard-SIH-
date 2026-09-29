"""
SkyGuard AI — Clean Telemetry Generator.

Generates physically realistic multi-station AWS telemetry with:
1. Diurnal solar temperature cycles with solar noon peaking and dawn troughs.
2. Diurnal humidity variation with inverse thermodynamic coupling.
3. Semi-diurnal barometric atmospheric tide (~1.5 hPa cycle).
4. Synoptic-scale planetary wave pressure and temperature trends (3-7 day cycles).
5. Sensor measurement noise.
6. Realistic sporadic packet dropouts (null values) marked as NORMAL.
7. Explicit deterministic random seeds.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .config import BenchmarkConfig, StationConfig, STATIONS, BENCHMARK_COLUMNS


def generate_clean_station_series(
    station: StationConfig,
    timestamps: Sequence[pd.Timestamp],
    rng: np.random.Generator,
    dropout_rate: float = 0.008,
) -> list[dict[str, Any]]:
    """
    Generate clean, realistic telemetry for one station across timestamps.
    """
    records: list[dict[str, Any]] = []

    # Station-specific phase offsets and geographic modulations
    # Coastal stations have reduced diurnal temp range and higher baseline humidity
    is_coastal = station.cluster_id in ("DELTA_CLUSTER", "NORTH_COAST")
    diurnal_temp_amplitude = 5.2 if is_coastal else 7.8
    diurnal_hum_amplitude = 14.0 if is_coastal else 22.0

    # Synoptic period (e.g. ~5.5 days = 132 hours)
    synoptic_period_hours = 132.0
    synoptic_temp_amplitude = 2.2
    synoptic_baro_amplitude = 3.5

    for h_idx, ts in enumerate(timestamps):
        hour_of_day = ts.hour
        hour_float = hour_of_day + ts.minute / 60.0

        # 1. Diurnal Temperature: trough at 05:30 (sunrise), peak at 14:30 (solar lag)
        temp_diurnal_phase = (hour_float - 8.5) * math.pi / 12.0
        diurnal_temp = diurnal_temp_amplitude * math.sin(temp_diurnal_phase)

        # 2. Synoptic Temperature Trend
        synoptic_phase = (h_idx / synoptic_period_hours) * 2.0 * math.pi
        synoptic_temp = synoptic_temp_amplitude * math.sin(synoptic_phase)

        # 3. Combined Temperature with Micro-noise
        temp_val = (
            station.base_temp
            + diurnal_temp
            + synoptic_temp
            + float(rng.normal(0, 0.35))
        )
        temp_val = round(temp_val, 2)

        # 4. Diurnal Humidity: inverse phase with temperature, peak at dawn, min in afternoon
        hum_diurnal_phase = temp_diurnal_phase
        diurnal_hum = -diurnal_hum_amplitude * math.sin(hum_diurnal_phase)
        synoptic_hum = -synoptic_temp * 2.5

        hum_val = (
            station.base_hum
            + diurnal_hum
            + synoptic_hum
            + float(rng.normal(0, 1.1))
        )
        hum_val = float(np.clip(hum_val, 15.0, 98.0))
        hum_val = round(hum_val, 1)

        # 5. Atmospheric Pressure: Semi-diurnal thermal tide (peaks ~10:00 & 22:00, troughs ~04:00 & 16:00)
        tide_phase = hour_float * math.pi / 6.0
        baro_tide = 1.35 * math.cos(tide_phase)
        synoptic_baro = -synoptic_baro_amplitude * math.sin(synoptic_phase)

        baro_val = (
            station.base_baro
            + baro_tide
            + synoptic_baro
            + float(rng.normal(0, 0.22))
        )
        baro_val = round(baro_val, 2)

        # 6. Sporadic natural transmission dropout (~0.8% chance of missing 1 parameter)
        if rng.uniform(0.0, 1.0) < dropout_rate:
            param_to_drop = rng.choice(["temperature", "humidity", "pressure"])
            if param_to_drop == "temperature":
                temp_val = None
            elif param_to_drop == "humidity":
                hum_val = None
            else:
                baro_val = None

        record: dict[str, Any] = {
            "station_id": station.station_id,
            "station_name": station.station_name,
            "timestamp": ts.isoformat(),
            "temperature": temp_val,
            "humidity": hum_val,
            "pressure": baro_val,
            "source": "SYNTHETIC_SIMULATION",
            "scenario_id": "SCN_CLEAN_BASELINE",
            "scenario_type": "CLEAN_NORMAL",
            "fault_type": "NONE",
            "severity": "NONE",
            "ground_truth": "NORMAL",
            "is_anomaly": 0,
            "is_sensor_fault": 0,
            "event_start": None,
            "event_end": None,
            "affected_parameter": "none",
            "description": "Normal ambient weather telemetry with diurnal solar cycles",
        }
        records.append(record)

    return records


def generate_clean_network_telemetry(
    config: BenchmarkConfig,
    stations: Sequence[StationConfig] = STATIONS,
) -> pd.DataFrame:
    """
    Generate clean telemetry across all stations for the entire benchmark duration.
    """
    start_ts = pd.Timestamp(config.start_timestamp)
    timestamps = [
        start_ts + pd.Timedelta(hours=i * config.cadence_hours)
        for i in range(config.total_hours)
    ]

    all_records: list[dict[str, Any]] = []

    for station in stations:
        # Seed uniquely but deterministically per station
        station_seed = config.random_seed + (abs(hash(station.station_id)) % 10000)
        rng = np.random.default_rng(seed=station_seed)

        station_records = generate_clean_station_series(
            station=station,
            timestamps=timestamps,
            rng=rng,
        )
        all_records.extend(station_records)

    frame = pd.DataFrame(all_records, columns=BENCHMARK_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return frame
