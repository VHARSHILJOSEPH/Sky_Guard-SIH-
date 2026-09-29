from __future__ import annotations

from typing import Any


def build_reason_codes(
    anomaly_type: str,
    historical_score: float,
    temporal_score: float,
    multivariate_score: float,
    spatial_score: float | None,
    forecast_score: float | None,
    quality_flags: list[str],
    ml_score: float | None = None,
    threshold: float = 0.40,
) -> list[str]:
    codes: list[str] = []

    if anomaly_type not in ("NORMAL", "NONE", "UNKNOWN_ANOMALY"):
        codes.append(anomaly_type)

    if ml_score is not None and ml_score >= threshold:
        codes.append("ML_ANOMALY_TRIGGERED")

    if historical_score >= threshold:
        codes.append("HISTORICAL_DEVIATION")

    if temporal_score >= threshold:
        codes.append("TEMPORAL_RATE_LIMIT")

    if multivariate_score >= threshold:
        codes.append("MULTIVARIATE_INCONSISTENCY")

    if spatial_score is not None and spatial_score >= 0.50:
        codes.append("SPATIAL_DISAGREEMENT")

    if forecast_score is not None and forecast_score >= 0.50:
        codes.append("FORECAST_INCONSISTENCY")

    for flag in quality_flags:
        if flag not in codes:
            codes.append(flag)

    return codes


def build_structured_explanation(
    status: str,
    anomaly_type: str,
    evidence: dict[str, Any],
    reason_codes: list[str],
    event_classification: str,
    observation: dict[str, Any] | None = None,
    deltas: dict[str, float | None] | None = None,
    affected_parameter: str = "unknown",
    threshold: float = 0.40,
) -> dict[str, Any]:
    """
    Answers the 5 core diagnostic questions to provide transparent, honest explainability:
    1. What happened?
    2. Why was it flagged?
    3. What evidence supports this?
    4. What evidence contradicts this?
    5. What evidence is missing or stale?
    """
    # 1. What Happened?
    if status == "NORMAL" or anomaly_type == "NORMAL":
        what_happened = (
            "All meteorological observations and rate-of-change metrics remain within expected "
            "climatological and sensor operational limits."
        )
    elif anomaly_type == "COMMUNICATION_FAILURE":
        what_happened = "Telemetry stream interruption or empty payload received with no valid sensor data."
    elif anomaly_type == "DATA_QUALITY_ANOMALY":
        what_happened = (
            f"Physical bounds violation or data corruption detected in {affected_parameter} telemetry."
        )
    elif anomaly_type == "FROZEN_SENSOR":
        what_happened = (
            f"Sensor transducer in {affected_parameter} is stuck, exhibiting zero variability over consecutive intervals."
        )
    elif anomaly_type == "SENSOR_DRIFT":
        what_happened = (
            f"Systematic baseline drift detected in {affected_parameter} departing from historical diurnal expectations."
        )
    elif anomaly_type == "TEMPERATURE_ANOMALY":
        dt = deltas.get("temperature") if deltas else None
        delta_str = f" ({dt:+.1f}°C)" if dt is not None else ""
        what_happened = f"Abrupt temperature excursion detected{delta_str} exceeding normal physical rate of change."
    elif anomaly_type == "HUMIDITY_ANOMALY":
        dh = deltas.get("humidity") if deltas else None
        delta_str = f" ({dh:+.1f}%)" if dh is not None else ""
        what_happened = f"Abrupt relative humidity excursion detected{delta_str} inconsistent with local atmosphere."
    elif anomaly_type == "PRESSURE_ANOMALY":
        dp = deltas.get("pressure") if deltas else None
        delta_str = f" ({dp:+.1f} hPa)" if dp is not None else ""
        what_happened = f"Sharp atmospheric pressure deviation detected{delta_str} outside expected barometric range."
    elif anomaly_type == "MULTIVARIATE_INCONSISTENCY":
        what_happened = (
            "Inter-variable relationship violation: temperature, humidity, and pressure decouple "
            "from expected atmospheric thermodynamic relationships."
        )
    elif anomaly_type == "LIKELY_WEATHER_EVENT":
        what_happened = (
            "Regional atmospheric disturbance or severe weather onset: coordinated rapid changes "
            "corroborated by peer stations or thermodynamic coupling."
        )
    elif anomaly_type == "INSUFFICIENT_EVIDENCE":
        what_happened = "Inconclusive telemetry pattern with contradictory or sparse evidence channels."
    else:
        what_happened = f"Station observation flagged with anomalous condition: {anomaly_type}."

    # 2. Why Flagged?
    ml_score = evidence.get("ml_score", 0.0)
    temp_score = evidence.get("temporal_score", 0.0)
    hist_score = evidence.get("historical_score", 0.0)
    multi_score = evidence.get("multivariate_score", 0.0)
    quality_score = evidence.get("data_quality_score", 0.0)

    trigger_reasons = []
    if quality_score > 0.1:
        trigger_reasons.append(f"Data quality penalty ({quality_score:.2f})")
    if ml_score >= threshold:
        trigger_reasons.append(f"ML detector score ({ml_score:.2f}) exceeded threshold ({threshold:.2f})")
    if temp_score >= threshold:
        trigger_reasons.append(f"Temporal rate score ({temp_score:.2f}) exceeded threshold ({threshold:.2f})")
    if hist_score >= threshold:
        trigger_reasons.append(f"Historical deviation score ({hist_score:.2f}) exceeded threshold ({threshold:.2f})")
    if multi_score >= threshold:
        trigger_reasons.append(f"Multivariate deviation score ({multi_score:.2f}) exceeded threshold ({threshold:.2f})")

    why_flagged = (
        "; ".join(trigger_reasons)
        if trigger_reasons
        else f"Observation classified as {status} based on multi-channel evidence synthesis."
    )

    # 3. Supporting Evidence
    supporting = []
    if ml_score >= threshold:
        supporting.append({"channel": "ML", "metric": "LOF Anomaly Score", "value": round(ml_score, 4)})
    if temp_score >= threshold:
        supporting.append({"channel": "Temporal", "metric": "Rate of Change", "value": round(temp_score, 4)})
    if hist_score >= threshold:
        supporting.append({"channel": "Statistical", "metric": "Historical Z-Score", "value": round(hist_score, 4)})
    if multi_score >= threshold:
        supporting.append({"channel": "Multivariate", "metric": "Covariance Deviation", "value": round(multi_score, 4)})

    spatial_info = evidence.get("spatial", {})
    if spatial_info and spatial_info.get("status") == "SPATIAL_NOT_CORROBORATED":
        supporting.append({"channel": "Spatial", "metric": "Peer Disagreement", "value": spatial_info.get("spatial_score")})
    elif spatial_info and spatial_info.get("status") == "SPATIAL_CORROBORATED" and anomaly_type == "LIKELY_WEATHER_EVENT":
        supporting.append({"channel": "Spatial", "metric": "Peer Corroboration", "value": spatial_info.get("spatial_agreement_score")})

    # 4. Contradicting Evidence
    contradicting = []
    if ml_score < threshold:
        contradicting.append({"channel": "ML", "metric": "LOF Anomaly Score", "value": round(ml_score, 4), "note": "Within normal cluster bounds"})
    if temp_score < 0.25:
        contradicting.append({"channel": "Temporal", "metric": "Rate of Change", "value": round(temp_score, 4), "note": "Nominal rate of change"})
    if hist_score < 0.25:
        contradicting.append({"channel": "Statistical", "metric": "Historical Baseline", "value": round(hist_score, 4), "note": "Matches historical distribution"})
    if spatial_info and spatial_info.get("status") == "SPATIAL_CORROBORATED" and anomaly_type != "LIKELY_WEATHER_EVENT":
        contradicting.append({"channel": "Spatial", "metric": "Peer Agreement", "value": spatial_info.get("spatial_agreement_score"), "note": "Nearby stations do not support isolated fault"})

    # 5. Missing or Stale Evidence
    missing = []
    if not spatial_info or spatial_info.get("status") == "SPATIAL_EVIDENCE_UNAVAILABLE":
        missing.append({"channel": "Spatial", "reason": "No active peer stations within 100km radius or timestamps stale"})
    forecast_info = evidence.get("forecast", {})
    if not forecast_info or forecast_info.get("status") == "UNAVAILABLE":
        details = forecast_info.get("details", {}) if forecast_info else {}
        missing.append({"channel": "Forecast", "reason": details.get("reason", "External forecast unavailable or timestamp stale")})

    return {
        "what_happened": what_happened,
        "why_flagged": why_flagged,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "missing_evidence": missing,
        "event_classification": event_classification,
    }


def build_explanation(
    status: str,
    anomaly_type: str,
    evidence: dict[str, Any],
    reason_codes: list[str],
    event_classification: str,
    observation: dict[str, Any] | None = None,
    deltas: dict[str, float | None] | None = None,
    affected_parameter: str = "unknown",
) -> str:
    """
    Builds a concise summary string synthesized from the structured 5-question diagnostics.
    """
    diag = build_structured_explanation(
        status=status,
        anomaly_type=anomaly_type,
        evidence=evidence,
        reason_codes=reason_codes,
        event_classification=event_classification,
        observation=observation,
        deltas=deltas,
        affected_parameter=affected_parameter,
    )

    statements = [diag["what_happened"], diag["why_flagged"]]
    if event_classification and event_classification != "NORMAL":
        statements.append(f"Classification: {event_classification}.")

    return " ".join(statements)


def recommended_action(
    status: str,
    event_classification: str,
    anomaly_type: str,
    affected_parameter: str = "unknown",
) -> str:
    if anomaly_type in {
        "MISSING_DATA",
        "COMMUNICATION_FAILURE",
        "DUPLICATE_DATA",
        "TIMESTAMP_ERROR",
    }:
        return "Check AWS telemetry transmission modem, antenna line, and NTP timestamp synchronization."

    if anomaly_type == "DATA_QUALITY_ANOMALY":
        return f"Verify physical transducer wiring and analog-to-digital converter for {affected_parameter} sensor."

    if anomaly_type == "FROZEN_SENSOR":
        return f"Power-cycle transducer and inspect physical mounting for {affected_parameter} sensor freeze."

    if anomaly_type == "SENSOR_DRIFT":
        return f"Schedule on-site calibration verification for {affected_parameter} against an IMD reference standard."

    if event_classification == "LIKELY_WEATHER_EVENT":
        return "No sensor repair needed. Meteorological event confirmed by spatial network; flag for weather monitoring."

    if event_classification == "LIKELY_SENSOR_ANOMALY":
        return f"Inspect and test {affected_parameter} sensor hardware; isolated departure uncorroborated by peer stations."

    if status == "CRITICAL":
        return "Escalate immediately to maintenance dispatch for station hardware inspection."

    if status == "WARNING":
        return "Continue automated monitoring; flag for maintenance if anomaly persists across subsequent observations."

    return "Nominal operation; no immediate action required."
