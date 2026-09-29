"""
SkyGuard AI — Synthetic Fault Generator.

Generates realistic sensor faults, data quality issues, and communication failures
across three distinct severity levels: MILD, MODERATE, and SEVERE.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Sequence
import numpy as np
import pandas as pd

from .config import FAULT_TYPES, SeverityLevel, GroundTruthType


def get_ground_truth_category(fault_type: str) -> GroundTruthType:
    """Map fault type to high-level ground truth category."""
    if fault_type in (
        "TEMPERATURE_SPIKE",
        "TEMPERATURE_DROP",
        "TEMPERATURE_DRIFT",
        "HUMIDITY_SPIKE",
        "HUMIDITY_DRIFT",
        "PRESSURE_SPIKE",
        "PRESSURE_DRIFT",
        "FROZEN_SENSOR",
        "MULTIVARIATE_INCONSISTENCY",
    ):
        return "SENSOR_FAULT"
    elif fault_type in ("MISSING_DATA", "DUPLICATE_DATA", "TIMESTAMP_ERROR"):
        return "DATA_QUALITY_ISSUE"
    elif fault_type in ("COMMUNICATION_FAILURE",):
        return "COMMUNICATION_FAULT"
    return "NORMAL"


def inject_fault_into_slice(
    clean_slice: list[dict[str, Any]],
    fault_type: str,
    severity: SeverityLevel,
    scenario_id: str,
    rng: np.random.Generator,
    start_offset: int = 4,
    duration: int = 6,
) -> list[dict[str, Any]]:
    """
    Inject a synthetic fault into a contiguous window of station records.
    Returns the modified slice with complete ground truth metadata.
    """
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    result = [copy.deepcopy(r) for r in clean_slice]
    n_records = len(result)
    start_idx = max(0, min(start_offset, n_records - 1))
    end_idx = min(start_idx + duration, n_records)

    event_start_ts = result[start_idx]["timestamp"]
    event_end_ts = result[end_idx - 1]["timestamp"]
    gt_category = get_ground_truth_category(fault_type)

    affected_param = "none"

    if fault_type == "TEMPERATURE_SPIKE":
        affected_param = "temperature"
        mag = 3.0 if severity == "MILD" else (7.2 if severity == "MODERATE" else 15.0)
        mag += float(rng.uniform(-0.4, 0.4))
        for i in range(start_idx, end_idx):
            if result[i]["temperature"] is not None:
                result[i]["temperature"] = round(result[i]["temperature"] + mag, 2)

    elif fault_type == "TEMPERATURE_DROP":
        affected_param = "temperature"
        mag = 3.0 if severity == "MILD" else (7.2 if severity == "MODERATE" else 15.0)
        mag += float(rng.uniform(-0.4, 0.4))
        for i in range(start_idx, end_idx):
            if result[i]["temperature"] is not None:
                result[i]["temperature"] = round(result[i]["temperature"] - mag, 2)

    elif fault_type == "TEMPERATURE_DRIFT":
        affected_param = "temperature"
        rate = 0.28 if severity == "MILD" else (0.75 if severity == "MODERATE" else 1.85)
        for offset, i in enumerate(range(start_idx, end_idx), start=1):
            if result[i]["temperature"] is not None:
                result[i]["temperature"] = round(result[i]["temperature"] + rate * offset, 2)

    elif fault_type == "HUMIDITY_SPIKE":
        affected_param = "humidity"
        mag = 14.0 if severity == "MILD" else (26.0 if severity == "MODERATE" else 45.0)
        for i in range(start_idx, end_idx):
            if result[i]["humidity"] is not None:
                val = float(np.clip(result[i]["humidity"] + mag, 0.0, 100.0))
                result[i]["humidity"] = round(val, 1)

    elif fault_type == "HUMIDITY_DRIFT":
        affected_param = "humidity"
        rate = -0.7 if severity == "MILD" else (-1.8 if severity == "MODERATE" else -4.2)
        for offset, i in enumerate(range(start_idx, end_idx), start=1):
            if result[i]["humidity"] is not None:
                val = float(np.clip(result[i]["humidity"] + rate * offset, 5.0, 100.0))
                result[i]["humidity"] = round(val, 1)

    elif fault_type == "PRESSURE_SPIKE":
        affected_param = "pressure"
        mag = 3.8 if severity == "MILD" else (10.5 if severity == "MODERATE" else 24.0)
        for i in range(start_idx, end_idx):
            if result[i]["pressure"] is not None:
                result[i]["pressure"] = round(result[i]["pressure"] + mag, 2)

    elif fault_type == "PRESSURE_DRIFT":
        affected_param = "pressure"
        rate = 0.35 if severity == "MILD" else (0.95 if severity == "MODERATE" else 2.6)
        for offset, i in enumerate(range(start_idx, end_idx), start=1):
            if result[i]["pressure"] is not None:
                result[i]["pressure"] = round(result[i]["pressure"] + rate * offset, 2)

    elif fault_type == "FROZEN_SENSOR":
        freeze_src = max(0, start_idx - 1)
        if severity == "MILD":
            affected_param = "temperature"
            frozen_val = result[freeze_src]["temperature"]
            for i in range(start_idx, end_idx):
                result[i]["temperature"] = frozen_val
        elif severity == "MODERATE":
            affected_param = "temperature,humidity"
            f_temp = result[freeze_src]["temperature"]
            f_hum = result[freeze_src]["humidity"]
            for i in range(start_idx, end_idx):
                result[i]["temperature"] = f_temp
                result[i]["humidity"] = f_hum
        else:
            affected_param = "all"
            f_temp = result[freeze_src]["temperature"]
            f_hum = result[freeze_src]["humidity"]
            f_baro = result[freeze_src]["pressure"]
            for i in range(start_idx, end_idx):
                result[i]["temperature"] = f_temp
                result[i]["humidity"] = f_hum
                result[i]["pressure"] = f_baro

    elif fault_type == "MISSING_DATA":
        if severity == "MILD":
            affected_param = "temperature"
            for i in range(start_idx, end_idx):
                result[i]["temperature"] = None
        else:
            affected_param = "all"
            for i in range(start_idx, end_idx):
                result[i]["temperature"] = None
                result[i]["humidity"] = None
                result[i]["pressure"] = None

    elif fault_type == "DUPLICATE_DATA":
        affected_param = "all"
        dup_src = max(0, start_idx - 1)
        dup_row = copy.deepcopy(result[dup_src])
        for i in range(start_idx, end_idx):
            result[i]["temperature"] = dup_row["temperature"]
            result[i]["humidity"] = dup_row["humidity"]
            result[i]["pressure"] = dup_row["pressure"]

    elif fault_type == "TIMESTAMP_ERROR":
        affected_param = "timestamp"
        if severity == "MILD":
            # 1-hour jitter
            for i in range(start_idx, end_idx):
                ts_obj = pd.Timestamp(result[i]["timestamp"]) - pd.Timedelta(hours=1)
                result[i]["timestamp"] = ts_obj.isoformat()
        elif severity == "MODERATE":
            # 1-day backward jump
            for i in range(start_idx, end_idx):
                ts_obj = pd.Timestamp(result[i]["timestamp"]) - pd.Timedelta(days=1)
                result[i]["timestamp"] = ts_obj.isoformat()
        else:
            # Epoch reset
            for i in range(start_idx, end_idx):
                result[i]["timestamp"] = "1970-01-01T00:00:00Z"

    elif fault_type == "COMMUNICATION_FAILURE":
        affected_param = "all"
        for i in range(start_idx, end_idx):
            result[i]["temperature"] = None
            result[i]["humidity"] = None
            result[i]["pressure"] = None

    elif fault_type == "MULTIVARIATE_INCONSISTENCY":
        affected_param = "multivariate"
        if severity == "MILD":
            # Subtle decoupling: midday temperature warming accompanied by unseasonal humidity surge
            for i in range(start_idx, end_idx):
                if result[i]["temperature"] is not None:
                    result[i]["temperature"] = round(result[i]["temperature"] + 3.2, 2)
                if result[i]["humidity"] is not None:
                    result[i]["humidity"] = round(min(95.0, result[i]["humidity"] + 18.0), 1)
        elif severity == "MODERATE":
            # 42C with 96% humidity
            for i in range(start_idx, end_idx):
                result[i]["temperature"] = 42.5
                result[i]["humidity"] = 96.0
        else:
            # Physically contradictory: 48C with 99% humidity and 840 hPa
            for i in range(start_idx, end_idx):
                result[i]["temperature"] = 48.8
                result[i]["humidity"] = 99.0
                result[i]["pressure"] = 840.0

    # Tag ground truth on faulted range
    for i in range(start_idx, end_idx):
        result[i]["scenario_id"] = scenario_id
        result[i]["scenario_type"] = "SINGLE_STATION_FAULT" if gt_category == "SENSOR_FAULT" else (
            "DATA_QUALITY" if gt_category == "DATA_QUALITY_ISSUE" else "COMMUNICATION"
        )
        result[i]["fault_type"] = fault_type
        result[i]["severity"] = severity
        result[i]["ground_truth"] = gt_category
        result[i]["is_anomaly"] = 1
        result[i]["is_sensor_fault"] = 1
        result[i]["event_start"] = event_start_ts
        result[i]["event_end"] = event_end_ts
        result[i]["affected_parameter"] = affected_param
        result[i]["description"] = f"Synthetic {severity} {fault_type.lower().replace('_', ' ')} injection"

    return result
