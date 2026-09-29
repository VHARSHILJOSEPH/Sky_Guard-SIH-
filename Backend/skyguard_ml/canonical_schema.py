"""
SkyGuard AI — Canonical Telemetry and Inference Schemas.

Defines:
1. CanonicalTelemetryInput: Strict input validation and alias normalization.
2. CanonicalEvidence: 7-channel evidence fusion data model.
3. CanonicalInferenceResponse: Authoritative structured inference contract.
4. AnomalyIncidentModel: Incident lifecycle and alert deduplication contract.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# Allowed Source Taxonomy
AllowedSource = Literal[
    "IMD_AWS",
    "DEMO_SIMULATION",
    "OPEN_METEO_CONTEXT",
    "OFFLINE_PREVIEW",
]

ALLOWED_SOURCES = (
    "IMD_AWS",
    "DEMO_SIMULATION",
    "OPEN_METEO_CONTEXT",
    "OFFLINE_PREVIEW",
)

# Canonical Classification States
ClassificationState = Literal[
    "NORMAL",
    "ANOMALOUS",
    "POSSIBLE_SENSOR_FAULT",
    "POSSIBLE_WEATHER_EVENT",
    "DATA_QUALITY_ISSUE",
    "COMMUNICATION_FAILURE",
    "WARMUP",
    "INSUFFICIENT_EVIDENCE",
]

# Anomaly Lifecycle States
IncidentLifecycleState = Literal[
    "DETECTED",
    "ACKNOWLEDGED",
    "INVESTIGATING",
    "RESOLVED",
]

# Severity Levels
SeverityLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _clean_numeric(v: Any) -> Optional[float]:
    """Helper to sanitize numeric values, rejecting NaN/Inf."""
    if v is None or v == "" or v == "NaN" or v == "null" or v == "None":
        return None
    try:
        val = float(v)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def parse_and_validate_timestamp(v: Any) -> str:
    """Ensure timestamp is valid, normalized to ISO-8601 UTC string."""
    if v is None or v == "" or str(v).lower() in ("nan", "none", "null", "nat"):
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if isinstance(v, (int, float)):
        # Epoch timestamp
        dt = datetime.fromtimestamp(v, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if isinstance(v, datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
        else:
            v = v.astimezone(timezone.utc)
        return v.strftime("%Y-%m-%dT%H:%M:%SZ")

    s = str(v).strip()
    try:
        # Parse using standard ISO or pandas
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        # Year sanity check (1990 - 2100)
        if not (1990 <= dt.year <= 2100):
            raise ValueError(f"Timestamp year {dt.year} outside valid meteorological range 1990-2100")
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as exc:
        raise ValueError(f"Invalid timestamp format: {v!r}. Expected ISO-8601 UTC string.") from exc


class CanonicalTelemetryInput(BaseModel):
    """
    Standardized canonical telemetry schema.
    Accepts standard fields and common client aliases.
    """
    station_id: str = Field(..., description="Unique station identifier")
    timestamp: str = Field(..., description="Observation timestamp in ISO UTC")
    temperature_c: Optional[float] = Field(None, description="Ambient air temperature in Celsius")
    humidity_pct: Optional[float] = Field(None, description="Relative humidity in percent (0-100)")
    pressure_hpa: Optional[float] = Field(None, description="Barometric surface pressure in hPa")
    wind_speed_ms: Optional[float] = Field(None, description="Wind speed in meters/second")
    wind_direction_deg: Optional[float] = Field(None, description="Wind direction in degrees (0-360)")
    rainfall_mm: Optional[float] = Field(None, description="Precipitation/rainfall in mm")
    source: str = Field("IMD_AWS", description="Data source identifier")
    device_id: Optional[str] = Field(None, description="Optional hardware device identifier")
    session_id: Optional[str] = Field(None, description="Optional isolation session ID for demo replay")

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        d = dict(data)

        # 1. Station ID
        if "station_id" not in d or not d["station_id"]:
            stn = d.get("stationId") or d.get("id") or d.get("stn_id")
            if stn:
                d["station_id"] = str(stn).strip()

        # 2. Timestamp
        raw_ts = d.get("timestamp") or d.get("time") or d.get("datetime")
        d["timestamp"] = parse_and_validate_timestamp(raw_ts)

        # 3. Temperature aliases
        temp_val = (
            d.get("temperature_c")
            if "temperature_c" in d
            else (d.get("temperature") if "temperature" in d else (d.get("temp") if "temp" in d else d.get("TA")))
        )
        d["temperature_c"] = _clean_numeric(temp_val)

        # 4. Humidity aliases
        hum_val = (
            d.get("humidity_pct")
            if "humidity_pct" in d
            else (d.get("humidity") if "humidity" in d else (d.get("hum") if "hum" in d else d.get("RH")))
        )
        d["humidity_pct"] = _clean_numeric(hum_val)

        # 5. Pressure aliases
        press_val = (
            d.get("pressure_hpa")
            if "pressure_hpa" in d
            else (d.get("pressure") if "pressure" in d else (d.get("baro") if "baro" in d else (d.get("PRES") or d.get("MSLP"))))
        )
        d["pressure_hpa"] = _clean_numeric(press_val)

        # 6. Wind aliases
        wind_val = (
            d.get("wind_speed_ms")
            if "wind_speed_ms" in d
            else (d.get("wind_speed") if "wind_speed" in d else (d.get("WS") if "WS" in d else d.get("wind")))
        )
        d["wind_speed_ms"] = _clean_numeric(wind_val)

        wind_dir = d.get("wind_direction_deg") or d.get("wind_direction") or d.get("WD")
        d["wind_direction_deg"] = _clean_numeric(wind_dir)

        # 7. Rainfall aliases
        rain_val = (
            d.get("rainfall_mm")
            if "rainfall_mm" in d
            else (d.get("rainfall") if "rainfall" in d else (d.get("RAIN") if "RAIN" in d else d.get("rain")))
        )
        d["rainfall_mm"] = _clean_numeric(rain_val)

        # 8. Source normalization
        raw_source = str(d.get("source") or "IMD_AWS").strip().upper()
        if raw_source in ("LIVE", "LIVE_AWS", "LIVE_IMD_STREAM", "IMD"):
            d["source"] = "IMD_AWS"
        elif raw_source in ("DEMO", "DEMO_REPLAY", "SYNTHETIC", "SIMULATION"):
            d["source"] = "DEMO_SIMULATION"
        elif raw_source in ("OPEN_METEO", "FORECAST", "OPENMETEO"):
            d["source"] = "OPEN_METEO_CONTEXT"
        elif raw_source in ("OFFLINE", "PREVIEW"):
            d["source"] = "OFFLINE_PREVIEW"
        else:
            d["source"] = raw_source

        return d

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        if v not in ALLOWED_SOURCES:
            raise ValueError(
                f"Invalid source: {v!r}. Allowed sources: {', '.join(ALLOWED_SOURCES)}"
            )
        return v


class CanonicalEvidence(BaseModel):
    """Structured 7-channel evidence fusion breakdown."""
    data_quality: dict[str, Any] = Field(default_factory=dict)
    statistical: dict[str, Any] = Field(default_factory=dict)
    temporal: dict[str, Any] = Field(default_factory=dict)
    ml: dict[str, Any] = Field(default_factory=dict)
    multivariate: dict[str, Any] = Field(default_factory=dict)
    spatial: dict[str, Any] = Field(default_factory=dict)
    weather_context: dict[str, Any] = Field(default_factory=dict)
    channels: dict[str, Any] = Field(default_factory=dict)
    has_conflict: bool = False
    conflict_details: Optional[str] = None
    is_override: bool = False
    override_reason: Optional[str] = None
    persistence: dict[str, Any] = Field(default_factory=dict)


class ModelMetadataModel(BaseModel):
    """Model identification and threshold metadata."""
    name: str
    version: str
    feature_schema_version: str
    threshold: Optional[float] = None
    calibrated: bool = True


class AnomalyIncidentModel(BaseModel):
    """Alert incident lifecycle and deduplication state."""
    incident_id: str
    status: IncidentLifecycleState
    first_detected: str
    last_detected: str
    occurrence_count: int = 1
    duration_seconds: float = 0.0
    anomaly_type: str
    severity: SeverityLevel


class CanonicalInferenceResponse(BaseModel):
    """
    Authoritative inference response returned by the backend.
    """
    station_id: str
    timestamp: str
    source: AllowedSource
    session_id: Optional[str] = None
    device_id: Optional[str] = None

    # Observed Telemetry
    readings: dict[str, Optional[float]]
    previous_readings: Optional[dict[str, Optional[float]]] = None
    deltas: Optional[dict[str, Optional[float]]] = None

    # Classification & Decisions
    final_status: ClassificationState
    is_anomaly: bool
    anomaly_type: str
    affected_parameter: str = "unknown"
    severity: SeverityLevel
    anomaly_score: float = Field(..., description="Fused anomaly score (0.0 to 1.0, not a probability)")
    evidence_strength: float = Field(..., description="Signal strength of primary anomalous evidence (0.0 to 1.0)")
    confidence_status: str = "NOT_CALIBRATED"

    # Model & Evidence
    model: ModelMetadataModel
    evidence: CanonicalEvidence
    explanation: dict[str, Any]
    structured_explanation: Optional[dict[str, Any]] = None
    evidence_waterfall: list[dict[str, Any]] = Field(default_factory=list)

    # Pipeline States
    warmup_state: dict[str, Any]
    data_quality_state: dict[str, Any]
    sensor_health: dict[str, Any]

    # Incident Management
    incident: Optional[AnomalyIncidentModel] = None

    # Performance
    inference_time_ms: float

