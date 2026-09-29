"""
SkyGuard AI — Weather Event Scenario Generator.

Generates physically plausible, multi-variable and multi-station meteorological events:
1. Convective Thunder-Storm Front: Rapid evaporative cooling, pressure jump, humidity surge.
2. Regional Heatwave Surge: Hot dry air mass advection, conserved mixing ratio, low RH.
3. Synoptic Low Pressure Trough: Deep barometric depression with suppressed diurnal range.
4. Coastal Sea Breeze Intrusion: Mesoscale maritime front with temperature drop and moisture jump.

The purpose is to benchmark whether SkyGuard correctly identifies genuine environmental
transitions instead of misclassifying them as hardware sensor failures.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from .config import WEATHER_EVENT_TYPES, GroundTruthType


def inject_convective_storm_front(
    records_by_station: dict[str, list[dict[str, Any]]],
    station_ids: Sequence[str],
    start_index: int,
    duration_hours: int = 4,
    scenario_id: str = "SCN_WX_CONVECTIVE_STORM_01",
    propagation_lags: Mapping[str, int] | None = None,
) -> None:
    """
    Inject a convective storm front across clustered stations.
    """
    propagation_lags = propagation_lags or {}

    for stn_id in station_ids:
        if stn_id not in records_by_station:
            continue

        records = records_by_station[stn_id]
        stn_lag = propagation_lags.get(stn_id, 0)
        s_idx = max(0, min(start_index + stn_lag, len(records) - 1))
        e_idx = min(s_idx + duration_hours, len(records))

        if e_idx <= s_idx:
            continue

        event_start_ts = records[s_idx]["timestamp"]
        event_end_ts = records[e_idx - 1]["timestamp"]

        for offset, i in enumerate(range(s_idx, e_idx)):
            # Storm dynamics:
            # - Hour 0: Rapid downdraft cooling (-7.5°C), humidity surge (+30%), pressure dip (-2.0 hPa)
            # - Hours 1-2: Rain cooling sustained, pressure rebounds (+2.5 hPa cold pool surge)
            # - Hours 3+: Gradual relaxation back to synoptic baseline
            decay = math.exp(-offset / 2.5)

            if records[i]["temperature"] is not None:
                cooling = -7.8 * decay
                records[i]["temperature"] = round(records[i]["temperature"] + cooling, 2)

            if records[i]["humidity"] is not None:
                hum_jump = 32.0 * decay
                records[i]["humidity"] = round(min(98.0, records[i]["humidity"] + hum_jump), 1)

            if records[i]["pressure"] is not None:
                # Gust front pressure jump / cold dome
                p_perturbation = (-1.8 if offset == 0 else 1.6) * decay
                records[i]["pressure"] = round(records[i]["pressure"] + p_perturbation, 2)

            records[i]["scenario_id"] = scenario_id
            records[i]["scenario_type"] = "MULTI_STATION_WEATHER_EVENT"
            records[i]["fault_type"] = "NONE"
            records[i]["severity"] = "MODERATE"
            records[i]["ground_truth"] = "WEATHER_EVENT"
            records[i]["is_anomaly"] = 1
            records[i]["is_sensor_fault"] = 0
            records[i]["event_start"] = event_start_ts
            records[i]["event_end"] = event_end_ts
            records[i]["affected_parameter"] = "all"
            records[i]["description"] = "Convective thunderstorm squall line passage across delta stations"


def inject_regional_heatwave(
    records_by_station: dict[str, list[dict[str, Any]]],
    station_ids: Sequence[str],
    start_index: int,
    duration_hours: int = 12,
    scenario_id: str = "SCN_WX_HEATWAVE_SURGE_01",
) -> None:
    """
    Inject a regional heatwave surge across inland stations.
    """
    for stn_id in station_ids:
        if stn_id not in records_by_station:
            continue

        records = records_by_station[stn_id]
        s_idx = max(0, min(start_index, len(records) - 1))
        e_idx = min(s_idx + duration_hours, len(records))

        if e_idx <= s_idx:
            continue

        event_start_ts = records[s_idx]["timestamp"]
        event_end_ts = records[e_idx - 1]["timestamp"]

        for offset, i in enumerate(range(s_idx, e_idx)):
            # Bell-shaped diurnal heat anomaly peaking at midday
            heat_factor = math.sin((offset / duration_hours) * math.pi)

            if records[i]["temperature"] is not None:
                temp_boost = 6.2 * heat_factor
                records[i]["temperature"] = round(records[i]["temperature"] + temp_boost, 2)

            if records[i]["humidity"] is not None:
                # Consistent thermodynamic drying (relative humidity drops as temp surges)
                hum_drop = -24.0 * heat_factor
                records[i]["humidity"] = round(max(15.0, records[i]["humidity"] + hum_drop), 1)

            if records[i]["pressure"] is not None:
                p_drop = -1.6 * heat_factor
                records[i]["pressure"] = round(records[i]["pressure"] + p_drop, 2)

            records[i]["scenario_id"] = scenario_id
            records[i]["scenario_type"] = "MULTI_STATION_WEATHER_EVENT"
            records[i]["fault_type"] = "NONE"
            records[i]["severity"] = "MODERATE"
            records[i]["ground_truth"] = "WEATHER_EVENT"
            records[i]["is_anomaly"] = 1
            records[i]["is_sensor_fault"] = 0
            records[i]["event_start"] = event_start_ts
            records[i]["event_end"] = event_end_ts
            records[i]["affected_parameter"] = "all"
            records[i]["description"] = "Regional heatwave advection with thermodynamic relative humidity drop"


def inject_synoptic_low_pressure(
    records_by_station: dict[str, list[dict[str, Any]]],
    station_ids: Sequence[str],
    start_index: int,
    duration_hours: int = 36,
    scenario_id: str = "SCN_WX_SYNOPTIC_LOW_01",
) -> None:
    """
    Inject a synoptic-scale tropical low pressure depression across the entire network.
    """
    for stn_id in station_ids:
        if stn_id not in records_by_station:
            continue

        records = records_by_station[stn_id]
        s_idx = max(0, min(start_index, len(records) - 1))
        e_idx = min(s_idx + duration_hours, len(records))

        if e_idx <= s_idx:
            continue

        event_start_ts = records[s_idx]["timestamp"]
        event_end_ts = records[e_idx - 1]["timestamp"]

        for offset, i in enumerate(range(s_idx, e_idx)):
            progress = offset / duration_hours
            trough_depth = math.sin(progress * math.pi)

            if records[i]["pressure"] is not None:
                p_drop = -9.8 * trough_depth
                records[i]["pressure"] = round(records[i]["pressure"] + p_drop, 2)

            if records[i]["humidity"] is not None:
                hum_lift = 18.0 * trough_depth
                records[i]["humidity"] = round(min(97.0, records[i]["humidity"] + hum_lift), 1)

            if records[i]["temperature"] is not None:
                # Dampened diurnal swing due to overcast skies
                records[i]["temperature"] = round(records[i]["temperature"] - 2.0 * trough_depth, 2)

            records[i]["scenario_id"] = scenario_id
            records[i]["scenario_type"] = "MULTI_STATION_WEATHER_EVENT"
            records[i]["fault_type"] = "NONE"
            records[i]["severity"] = "SEVERE"
            records[i]["ground_truth"] = "WEATHER_EVENT"
            records[i]["is_anomaly"] = 1
            records[i]["is_sensor_fault"] = 0
            records[i]["event_start"] = event_start_ts
            records[i]["event_end"] = event_end_ts
            records[i]["affected_parameter"] = "all"
            records[i]["description"] = "Synoptic low pressure depression passage over Bay of Bengal coastal AWS"


def inject_coastal_sea_breeze(
    records_by_station: dict[str, list[dict[str, Any]]],
    station_ids: Sequence[str],
    start_index: int,
    duration_hours: int = 6,
    scenario_id: str = "SCN_WX_SEA_BREEZE_01",
) -> None:
    """
    Inject a coastal sea-breeze front at maritime stations.
    """
    for stn_id in station_ids:
        if stn_id not in records_by_station:
            continue

        records = records_by_station[stn_id]
        s_idx = max(0, min(start_index, len(records) - 1))
        e_idx = min(s_idx + duration_hours, len(records))

        if e_idx <= s_idx:
            continue

        event_start_ts = records[s_idx]["timestamp"]
        event_end_ts = records[e_idx - 1]["timestamp"]

        for offset, i in enumerate(range(s_idx, e_idx)):
            factor = math.exp(-offset / 3.0)

            if records[i]["temperature"] is not None:
                t_cooling = -3.8 * factor
                records[i]["temperature"] = round(records[i]["temperature"] + t_cooling, 2)

            if records[i]["humidity"] is not None:
                h_surge = 21.0 * factor
                records[i]["humidity"] = round(min(96.0, records[i]["humidity"] + h_surge), 1)

            if records[i]["pressure"] is not None:
                p_lift = 1.1 * factor
                records[i]["pressure"] = round(records[i]["pressure"] + p_lift, 2)

            records[i]["scenario_id"] = scenario_id
            records[i]["scenario_type"] = "MULTI_STATION_WEATHER_EVENT"
            records[i]["fault_type"] = "NONE"
            records[i]["severity"] = "MILD"
            records[i]["ground_truth"] = "WEATHER_EVENT"
            records[i]["is_anomaly"] = 1
            records[i]["is_sensor_fault"] = 0
            records[i]["event_start"] = event_start_ts
            records[i]["event_end"] = event_end_ts
            records[i]["affected_parameter"] = "all"
            records[i]["description"] = "Afternoon coastal maritime sea-breeze front with cooling and moisture surge"
