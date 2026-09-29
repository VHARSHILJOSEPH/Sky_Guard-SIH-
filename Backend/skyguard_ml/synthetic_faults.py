from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FaultRecord:
    fault_type: str
    start_index: int
    duration: int
    magnitude: float
    direction: int
    severity: str = "moderate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "start_index": self.start_index,
            "duration": self.duration,
            "magnitude": self.magnitude,
            "direction": self.direction,
            "severity": self.severity,
        }


FAULT_TYPES = [
    "TEMPERATURE_SPIKE",
    "TEMPERATURE_DROP",
    "TEMPERATURE_DRIFT",
    "HUMIDITY_SPIKE",
    "HUMIDITY_DRIFT",
    "PRESSURE_SPIKE",
    "PRESSURE_DRIFT",
    "FROZEN_SENSOR",
    "MISSING_DATA",
    "COMMUNICATION_FAILURE",
    "DUPLICATE_DATA",
    "TIMESTAMP_ERROR",
    "MULTIVARIATE_INCONSISTENCY",
    "PHYSICAL_LIMIT_VIOLATION",
    "COORDINATED_WEATHER_EVENT",  # Benign scenario: target=0
]


def _valid_start(
    length: int,
    duration: int,
    rng: np.random.Generator,
) -> int:
    if length <= duration + 2:
        return 0
    return int(rng.integers(1, length - duration - 1))


def _choose_station_frame(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    stations = frame["station_id"].dropna().unique()

    if len(stations) == 0:
        return frame.copy()

    station = rng.choice(stations)
    result = frame[frame["station_id"] == station].copy()

    return result.sort_values("timestamp").reset_index(drop=True)


def generate_synthetic_faults(
    clean_frame: pd.DataFrame,
    faults_per_type: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    generated_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []

    severities = ["subtle", "moderate", "severe"]
    severity_multipliers = {"subtle": 0.5, "moderate": 1.0, "severe": 2.2}

    for fault_type in FAULT_TYPES:
        for fault_idx in range(faults_per_type):
            frame = _choose_station_frame(clean_frame, rng)

            if len(frame) < 8:
                continue

            severity = severities[fault_idx % len(severities)]
            sev_mult = severity_multipliers[severity]

            duration = int(
                rng.integers(
                    1,
                    max(2, min(12, len(frame) // 3)),
                )
            )
            start = _valid_start(len(frame), duration, rng)
            direction = int(rng.choice([-1, 1]))

            modified = frame.copy()
            orig_station = str(frame["station_id"].iloc[0])
            episode_id = f"{orig_station}_syn_{fault_type}_{fault_idx}"
            modified["station_id"] = episode_id
            indices = modified.index[start : start + duration]

            is_anomaly = True

            if fault_type == "TEMPERATURE_SPIKE":
                mag = rng.uniform(2.5, 6.0) * sev_mult
                modified.loc[indices, "temperature"] += mag

            elif fault_type == "TEMPERATURE_DROP":
                mag = rng.uniform(2.5, 6.0) * sev_mult
                modified.loc[indices, "temperature"] -= mag

            elif fault_type == "TEMPERATURE_DRIFT":
                mag = rng.uniform(0.15, 0.6) * sev_mult
                modified.loc[indices, "temperature"] += (
                    np.arange(duration) * mag * direction
                )

            elif fault_type == "HUMIDITY_SPIKE":
                mag = rng.uniform(10.0, 25.0) * sev_mult
                modified.loc[indices, "humidity"] = np.clip(
                    modified.loc[indices, "humidity"] + mag, 0.0, 100.0
                )

            elif fault_type == "HUMIDITY_DRIFT":
                mag = rng.uniform(0.8, 2.5) * sev_mult
                modified.loc[indices, "humidity"] = np.clip(
                    modified.loc[indices, "humidity"]
                    + np.arange(duration) * mag * direction,
                    0.0,
                    100.0,
                )

            elif fault_type == "PRESSURE_SPIKE":
                mag = rng.uniform(3.0, 12.0) * sev_mult
                modified.loc[indices, "pressure"] += mag

            elif fault_type == "PRESSURE_DRIFT":
                mag = rng.uniform(0.3, 1.2) * sev_mult
                modified.loc[indices, "pressure"] += (
                    np.arange(duration) * mag * direction
                )

            elif fault_type == "FROZEN_SENSOR":
                mag = 0.0
                target_var = rng.choice(["temperature", "humidity", "pressure"])
                if start > 0:
                    val = modified.loc[start - 1, target_var]
                    modified.loc[indices, target_var] = val
                else:
                    val = modified.loc[indices[0], target_var]
                    modified.loc[indices, target_var] = val

            elif fault_type == "MISSING_DATA":
                mag = 0.0
                modified.loc[indices, "temperature"] = np.nan
                modified.loc[indices, "humidity"] = np.nan

            elif fault_type == "COMMUNICATION_FAILURE":
                mag = 0.0
                modified = modified.drop(index=indices).reset_index(drop=True)

            elif fault_type == "DUPLICATE_DATA":
                mag = 0.0
                duplicate_rows = modified.loc[indices].copy()
                modified = pd.concat(
                    [
                        modified.iloc[:start],
                        duplicate_rows,
                        modified.iloc[start:],
                    ],
                    ignore_index=True,
                )

            elif fault_type == "TIMESTAMP_ERROR":
                mag = 0.0
                if len(indices) > 0:
                    modified.loc[indices[0], "timestamp"] = pd.Timestamp(
                        "1900-01-01", tz="UTC"
                    )

            elif fault_type == "MULTIVARIATE_INCONSISTENCY":
                # Temperature rises sharply while humidity also jumps unnaturally (violates psychrometric inverse correlation)
                mag = rng.uniform(4.0, 10.0) * sev_mult
                modified.loc[indices, "temperature"] += mag
                modified.loc[indices, "humidity"] = np.clip(
                    modified.loc[indices, "humidity"] + rng.uniform(15.0, 30.0),
                    0.0,
                    100.0,
                )

            elif fault_type == "PHYSICAL_LIMIT_VIOLATION":
                mag = 999.0
                target_var = rng.choice(["temperature", "humidity", "pressure"])
                if target_var == "humidity":
                    modified.loc[indices, "humidity"] = 125.0
                elif target_var == "temperature":
                    modified.loc[indices, "temperature"] = 75.0
                else:
                    modified.loc[indices, "pressure"] = 500.0

            elif fault_type == "COORDINATED_WEATHER_EVENT":
                # Benign frontal passage / thunderstorm: temp drops 3-6C, RH rises 20-30%, pressure dips 2-4hPa.
                # Target MUST remain 0 (normal) to test specificity against legitimate meteorological phenomena.
                is_anomaly = False
                mag = rng.uniform(3.0, 6.0)
                modified.loc[indices, "temperature"] -= mag
                modified.loc[indices, "humidity"] = np.clip(
                    modified.loc[indices, "humidity"] + rng.uniform(15.0, 30.0),
                    0.0,
                    100.0,
                )
                modified.loc[indices, "pressure"] -= rng.uniform(1.5, 4.0)

            modified["target"] = 0
            modified["fault_type"] = "NORMAL"

            if is_anomaly and len(modified) > 0:
                affected_start = min(start, len(modified) - 1)
                affected_end = min(
                    affected_start + duration,
                    len(modified),
                )
                modified.loc[
                    modified.index[affected_start:affected_end],
                    "target",
                ] = 1
                modified.loc[
                    modified.index[affected_start:affected_end],
                    "fault_type",
                ] = fault_type

            generated_frames.append(modified)
            metadata.append(
                FaultRecord(
                    fault_type=fault_type,
                    start_index=start,
                    duration=duration,
                    magnitude=float(mag) if "mag" in locals() else 1.0,
                    direction=direction,
                    severity=severity,
                ).as_dict()
            )

    if not generated_frames:
        return clean_frame.copy(), pd.DataFrame()

    generated = pd.concat(generated_frames, ignore_index=True)
    return generated, pd.DataFrame(metadata)
