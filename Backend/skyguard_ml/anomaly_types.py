from __future__ import annotations

from typing import Any

CANONICAL_ANOMALY_STATES = [
    "NORMAL",
    "TEMPERATURE_ANOMALY",
    "PRESSURE_ANOMALY",
    "HUMIDITY_ANOMALY",
    "FROZEN_SENSOR",
    "SENSOR_DRIFT",
    "DATA_QUALITY_ANOMALY",
    "COMMUNICATION_FAILURE",
    "MULTIVARIATE_INCONSISTENCY",
    "LIKELY_SENSOR_ANOMALY",
    "LIKELY_WEATHER_EVENT",
    "INSUFFICIENT_EVIDENCE",
    "WARMUP",
]


def infer_affected_parameter(
    anomaly_type: str,
    statistical: dict[str, Any] | None = None,
    quality_flags: list[str] | None = None,
    deltas: dict[str, float | None] | None = None,
) -> str:
    """
    Identifies the primary physical or digital parameter impacted by an anomaly.
    Returns: 'temperature' | 'pressure' | 'humidity' | 'multiple' | 'data' | 'communication' | 'unknown'
    """
    flags = set(quality_flags or [])
    stat = statistical or {}

    # 1. Communication & transmission failures
    if anomaly_type == "COMMUNICATION_FAILURE" or "COMMUNICATION_FAILURE" in flags:
        return "communication"

    # 2. Direct single-parameter anomaly types
    if anomaly_type == "TEMPERATURE_ANOMALY":
        return "temperature"
    if anomaly_type == "HUMIDITY_ANOMALY":
        return "humidity"
    if anomaly_type == "PRESSURE_ANOMALY":
        return "pressure"

    # 3. Multivariate or multi-sensor events
    if anomaly_type in ("MULTIVARIATE_INCONSISTENCY", "LIKELY_WEATHER_EVENT"):
        return "multiple"

    # 4. Frozen sensor examination
    frozen_stat = stat.get("frozen", {})
    frozen_vars = [v for v, info in frozen_stat.items() if info.get("score", 0.0) >= 0.6]
    if len(frozen_vars) == 1:
        return frozen_vars[0].lower()
    elif len(frozen_vars) > 1:
        return "multiple"

    # 5. Drift detector examination
    drift_stat = stat.get("drift", {})
    drift_vars = [v for v, info in drift_stat.items() if info.get("score", 0.0) >= 0.5]
    if len(drift_vars) == 1:
        return drift_vars[0].lower()
    elif len(drift_vars) > 1:
        return "multiple"

    # 6. Spike / Drop examination
    spike_stat = stat.get("spike_drop", {})
    spike_vars = [
        v for v, info in spike_stat.items()
        if info.get("spike_score", 0.0) >= 0.5 or info.get("drop_score", 0.0) >= 0.5
    ]
    if len(spike_vars) == 1:
        return spike_vars[0].lower()
    elif len(spike_vars) > 1:
        return "multiple"

    # 7. Quality flags inspection
    corrupt_vars = []
    if any("TEMPERATURE" in f for f in flags):
        corrupt_vars.append("temperature")
    if any("HUMIDITY" in f for f in flags):
        corrupt_vars.append("humidity")
    if any("PRESSURE" in f for f in flags):
        corrupt_vars.append("pressure")

    if len(corrupt_vars) == 1:
        return corrupt_vars[0]
    elif len(corrupt_vars) > 1:
        return "data"

    if any("TIMESTAMP" in f or "DUPLICATE" in f for f in flags):
        return "data"

    # 8. Check deltas for isolated excursion
    if deltas:
        candidates = []
        if deltas.get("temperature") is not None and abs(deltas["temperature"]) >= 4.0:
            candidates.append("temperature")
        if deltas.get("humidity") is not None and abs(deltas["humidity"]) >= 10.0:
            candidates.append("humidity")
        if deltas.get("pressure") is not None and abs(deltas["pressure"]) >= 3.0:
            candidates.append("pressure")

        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) > 1:
            return "multiple"

    if anomaly_type == "NORMAL":
        return "unknown"

    return "unknown"


def infer_anomaly_type(
    statistical: dict[str, Any],
    quality_flags: list[str],
    multivariate_score: float,
    is_warmup: bool = False,
    event_context: str | None = None,
    fused_score: float | None = None,
    threshold: float = 0.40,
) -> str:
    """
    Infers the authoritative classification state strictly adhering to
    the 13 Canonical Anomaly States.
    """
    flags = set(quality_flags)

    # 1. Warmup state precedence
    if is_warmup:
        return "WARMUP"

    # 2. Total communication failure precedence
    core_missing_flags = {
        "MISSING_TEMPERATURE",
        "MISSING_HUMIDITY",
        "MISSING_PRESSURE",
    }
    if core_missing_flags.issubset(flags) or "COMMUNICATION_FAILURE" in flags:
        return "COMMUNICATION_FAILURE"

    # 3. Data quality and corrupt telemetry precedence
    if (
        any("OUT_OF_RANGE" in flag for flag in flags)
        or any("TIMESTAMP" in flag for flag in flags)
        or any("DUPLICATE" in flag for flag in flags)
        or any(flag in core_missing_flags for flag in flags)
    ):
        return "DATA_QUALITY_ANOMALY"

    # 4. Frozen sensor check
    for variable, values in statistical.get("frozen", {}).items():
        if values.get("score", 0.0) >= 0.6:
            return "FROZEN_SENSOR"

    # 5. Sensor drift check
    for variable, values in statistical.get("drift", {}).items():
        if values.get("score", 0.0) >= 0.6:
            return "SENSOR_DRIFT"

    # 6. Spike / Drop / Isolated parameter check
    active_spikes: list[str] = []
    for variable, values in statistical.get("spike_drop", {}).items():
        if (
            values.get("spike_score", 0.0) >= 0.5
            or values.get("drop_score", 0.0) >= 0.5
        ):
            active_spikes.append(variable.lower())

    if len(active_spikes) > 1:
        return "MULTIVARIATE_INCONSISTENCY"
    elif len(active_spikes) == 1:
        var = active_spikes[0]
        if var == "temperature":
            return "TEMPERATURE_ANOMALY"
        elif var == "humidity":
            return "HUMIDITY_ANOMALY"
        elif var == "pressure":
            return "PRESSURE_ANOMALY"

    # 7. Weather event context
    if event_context == "LIKELY_WEATHER_EVENT":
        return "LIKELY_WEATHER_EVENT"

    if event_context == "LIKELY_SENSOR_ANOMALY":
        return "LIKELY_SENSOR_ANOMALY"

    # 8. Multivariate inconsistency
    if multivariate_score >= 0.60:
        return "MULTIVARIATE_INCONSISTENCY"

    # 9. Communication / data gap
    missing = statistical.get("missing", {})
    if missing.get("gap_count", 0) > 0:
        return "COMMUNICATION_FAILURE"
    if missing.get("missing_count", 0) > 0:
        return "DATA_QUALITY_ANOMALY"

    # 10. Check score threshold
    if fused_score is not None and fused_score < threshold:
        return "NORMAL"

    if event_context == "INSUFFICIENT_EVIDENCE":
        return "INSUFFICIENT_EVIDENCE"

    if fused_score is not None and fused_score >= threshold:
        return "LIKELY_SENSOR_ANOMALY"

    return "NORMAL"
