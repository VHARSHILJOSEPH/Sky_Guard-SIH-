from __future__ import annotations

import math
from typing import Any


def check_psychrometric_coherence(
    deltas: dict[str, float | None] | None,
) -> tuple[bool, str]:
    """
    Evaluates whether multi-parameter deltas follow known atmospheric thermodynamics
    (e.g., cold pool / gust front: temp drop + humidity rise + pressure rise/dip).
    """
    if not deltas:
        return False, "No delta context available"

    dt = deltas.get("temperature")
    dh = deltas.get("humidity")
    dp = deltas.get("pressure")

    # Frontal passage / thunderstorm gust front:
    # Rapid cooling combined with humidity surge
    if dt is not None and dh is not None:
        if dt <= -2.5 and dh >= 5.0:
            return True, f"Thermodynamic coherence: temperature drop ({dt:+.1f}°C) with humidity surge ({dh:+.1f}%)"
        if dt >= 3.0 and dh <= -8.0:
            return True, f"Diurnal/solar heating coherence: temperature rise ({dt:+.1f}°C) with relative humidity drop ({dh:+.1f}%)"

    # Pre-frontal pressure fall with rapid temperature change
    if dp is not None and dt is not None:
        if abs(dp) >= 2.0 and abs(dt) >= 2.0:
            return True, f"Synoptic pressure change ({dp:+.1f} hPa) aligned with temperature change ({dt:+.1f}°C)"

    return False, "Parameter shifts lack atmospheric coupling"


def check_isolated_sensor_fault(
    deltas: dict[str, float | None] | None,
) -> tuple[bool, str]:
    """
    Detects if a single parameter experienced an extreme change while
    all other atmospheric variables remained completely invariant/static.
    """
    if not deltas:
        return False, "No delta context available"

    dt = deltas.get("temperature")
    dh = deltas.get("humidity")
    dp = deltas.get("pressure")

    # Temperature spike while RH and Pressure remain virtually unchanged
    if dt is not None and abs(dt) >= 4.0:
        other_shifts = [abs(v) for v in (dh, dp) if v is not None]
        if other_shifts and max(other_shifts) < 1.5:
            return True, f"Isolated temperature excursion ({dt:+.1f}°C) with invariant humidity and pressure"

    # Humidity jump without temperature response
    if dh is not None and abs(dh) >= 15.0:
        other_shifts = [abs(v) for v in (dt, dp) if v is not None]
        if other_shifts and max(other_shifts) < 0.5:
            return True, f"Isolated humidity excursion ({dh:+.1f}%) with invariant temperature"

    # Pressure step change
    if dp is not None and abs(dp) >= 4.0:
        other_shifts = [abs(v) for v in (dt, dh) if v is not None]
        if other_shifts and max(other_shifts) < 1.0:
            return True, f"Isolated pressure step ({dp:+.1f} hPa) with invariant thermal conditions"

    return False, "No isolated parameter shift detected"


def classify_event_context(
    anomaly_score: float,
    historical_score: float,
    temporal_score: float,
    multivariate_score: float,
    spatial_score: float | None = None,
    forecast_score: float | None = None,
    observation: dict[str, Any] | None = None,
    deltas: dict[str, float | None] | None = None,
    spatial_status: str | None = None,
    has_conflict: bool = False,
) -> str:
    """
    Classifies anomalous signals into authoritative canonical event categories:
    1. LIKELY_WEATHER_EVENT
    2. LIKELY_SENSOR_ANOMALY
    3. INSUFFICIENT_EVIDENCE
    """
    # 1. If evidence is conflicted or anomaly score is trivial, classify INSUFFICIENT_EVIDENCE
    if has_conflict:
        return "INSUFFICIENT_EVIDENCE"

    if anomaly_score < 0.35 and historical_score < 0.35 and temporal_score < 0.35:
        return "INSUFFICIENT_EVIDENCE"

    # 2. Check physical / thermodynamic coupling in deltas
    is_coherent_weather, _ = check_psychrometric_coherence(deltas)
    is_isolated_sensor, _ = check_isolated_sensor_fault(deltas)

    # 3. Spatial peer corroboration status
    is_spatial_corroborated = (
        spatial_status == "SPATIAL_CORROBORATED"
        or (spatial_score is not None and spatial_score <= 0.30)
    )
    is_spatial_disagreed = (
        spatial_status == "SPATIAL_NOT_CORROBORATED"
        or (spatial_score is not None and spatial_score >= 0.55)
    )

    # 4. Synthesize sensor vs weather evidence weights
    sensor_evidence = (
        0.30 * anomaly_score
        + 0.25 * historical_score
        + 0.25 * temporal_score
        + 0.20 * multivariate_score
    )

    weather_sources: list[float] = []
    if spatial_score is not None:
        weather_sources.append(1.0 - spatial_score)
    if forecast_score is not None:
        weather_sources.append(1.0 - forecast_score)

    weather_evidence = (
        (sum(weather_sources) / len(weather_sources))
        if weather_sources
        else None
    )

    # 5. Rule evaluations in strict precedence
    # A. Isolated parameter shift with stable peers -> LIKELY_SENSOR_ANOMALY
    if is_isolated_sensor and not is_spatial_corroborated:
        return "LIKELY_SENSOR_ANOMALY"

    # B. Atmospheric thermodynamic coupling with peer agreement or neutral context -> LIKELY_WEATHER_EVENT
    if is_coherent_weather and not is_spatial_disagreed:
        return "LIKELY_WEATHER_EVENT"

    # C. Strong spatial corroboration confirming regional phenomenon
    if is_spatial_corroborated and weather_evidence is not None and weather_evidence >= 0.60:
        return "LIKELY_WEATHER_EVENT"

    # D. Strong spatial disagreement confirming localized instrument failure
    if is_spatial_disagreed and sensor_evidence >= 0.50:
        return "LIKELY_SENSOR_ANOMALY"

    # E. High sensor excursion in absence of external corroboration
    if weather_evidence is None:
        if sensor_evidence >= 0.65:
            return "LIKELY_SENSOR_ANOMALY"
        return "INSUFFICIENT_EVIDENCE"

    # F. Relative evidentiary balance
    if sensor_evidence >= 0.65 and weather_evidence <= 0.45:
        return "LIKELY_SENSOR_ANOMALY"

    if weather_evidence >= 0.65 and sensor_evidence <= 0.55:
        return "LIKELY_WEATHER_EVENT"

    return "INSUFFICIENT_EVIDENCE"
