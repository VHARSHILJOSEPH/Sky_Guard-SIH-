from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd


OBSERVATION_FIELDS = [
    "station_id",
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "latitude",
    "longitude",
]

NUMERIC_FIELDS = [
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "latitude",
    "longitude",
]


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def normalize_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "temp": "temperature",
        "temperature_c": "temperature",
        "relative_humidity": "humidity",
        "humidity_percent": "humidity",
        "humidity_pct": "humidity",
        "pressure_hpa": "pressure",
        "baro": "pressure",
        "barometer": "pressure",
        "slp": "pressure",
        "rain": "rainfall",
        "rainfall_mm": "rainfall",
        "wind": "wind_speed",
        "wind_speed_ms": "wind_speed",
        "wind_speed_kmh": "wind_speed",
    }

    normalized: dict[str, Any] = {}

    for key, value in observation.items():
        normalized[aliases.get(key, key)] = value

    normalized.setdefault("station_id", None)
    normalized.setdefault("timestamp", None)

    normalized["timestamp"] = normalize_timestamp(normalized["timestamp"])

    for field_name in NUMERIC_FIELDS:
        value = normalized.get(field_name)
        if value is None or value == "":
            normalized[field_name] = np.nan
            continue

        try:
            normalized[field_name] = float(value)
        except (TypeError, ValueError):
            normalized[field_name] = np.nan

    return {
        field_name: normalized.get(field_name, np.nan)
        for field_name in OBSERVATION_FIELDS
    }


def observations_to_frame(
    observations: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(observations, pd.DataFrame):
        records = observations.to_dict(orient="records")
    else:
        records = observations

    normalized = [normalize_observation(record) for record in records]
    frame = pd.DataFrame(normalized)

    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

    return frame


@dataclass
class ValidationReport:
    row_count: int
    valid_row_count: int
    invalid_row_count: int
    quality_flags: dict[int, list[str]] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        if self.row_count == 0:
            return 0.0
        return float(self.valid_row_count / self.row_count)


@dataclass
class ChannelEvidence:
    """Structured per-channel evidence model preserving traceability."""
    channel: str
    status: str  # AVAILABLE, UNAVAILABLE, STALE, NOT_APPLICABLE, INSUFFICIENT_HISTORY
    score: float | None = None
    weight: float = 0.0
    weighted_contribution: float = 0.0
    provenance: str = "AWS_TELEMETRY"
    timestamp: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "status": self.status,
            "score": round(self.score, 4) if self.score is not None else None,
            "weight": round(self.weight, 4),
            "weighted_contribution": round(self.weighted_contribution, 4),
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class DetectorEvidence:
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    status: str
    anomaly_score: float
    selected_model: str
    anomaly_type: str
    severity: str
    event_classification: str
    evidence: dict[str, Any]
    sensor_health: dict[str, Any]
    reason_codes: list[str]
    explanation: str
    recommended_action: str
    confidence_status: str = "NOT_CALIBRATED"
    affected_parameter: str = "unknown"
    structured_explanation: dict[str, Any] = field(default_factory=dict)
    evidence_waterfall: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "anomaly_score": round(float(self.anomaly_score), 4),
            "selected_model": self.selected_model,
            "anomaly_type": self.anomaly_type,
            "affected_parameter": self.affected_parameter,
            "severity": self.severity,
            "event_classification": self.event_classification,
            "evidence": self.evidence,
            "sensor_health": self.sensor_health,
            "reason_codes": self.reason_codes,
            "explanation": self.explanation,
            "structured_explanation": self.structured_explanation,
            "evidence_waterfall": self.evidence_waterfall,
            "recommended_action": self.recommended_action,
            "confidence_status": self.confidence_status,
        }
