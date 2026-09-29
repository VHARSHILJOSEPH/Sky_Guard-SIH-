from __future__ import annotations

from typing import Any
import math
import numpy as np

from .schemas import ChannelEvidence


def check_deterministic_overrides(
    observation: dict[str, Any] | None,
    quality_flags: list[str] | None,
) -> tuple[bool, str | None, str | None]:
    """
    Checks for non-negotiable physical or data integrity violations
    that bypass ML score weighting to prevent silent failure propagation.
    Returns: (is_override, override_type, reason)
    """
    flags = set(quality_flags or [])

    # 1. Physical bounds violations
    if observation:
        temp = observation.get("temperature")
        hum = observation.get("humidity")
        press = observation.get("pressure")

        if temp is not None and not (isinstance(temp, float) and math.isnan(temp)):
            try:
                t_val = float(temp)
                if t_val < -50.0 or t_val > 65.0:
                    return (
                        True,
                        "DATA_QUALITY_OVERRIDE",
                        f"Temperature ({t_val:.1f}°C) exceeds physical atmospheric bounds (-50°C to +65°C)",
                    )
            except (ValueError, TypeError):
                pass

        if hum is not None and not (isinstance(hum, float) and math.isnan(hum)):
            try:
                h_val = float(hum)
                if h_val < 0.0 or h_val > 100.0:
                    return (
                        True,
                        "DATA_QUALITY_OVERRIDE",
                        f"Relative humidity ({h_val:.1f}%) exceeds physical saturation bounds (0% to 100%)",
                    )
            except (ValueError, TypeError):
                pass

        if press is not None and not (isinstance(press, float) and math.isnan(press)):
            try:
                p_val = float(press)
                if p_val < 700.0 or p_val > 1150.0:
                    return (
                        True,
                        "DATA_QUALITY_OVERRIDE",
                        f"Atmospheric pressure ({p_val:.1f} hPa) exceeds terrestrial surface bounds (700 to 1150 hPa)",
                    )
            except (ValueError, TypeError):
                pass

        # 2. Total missing payload
        all_null = True
        for k in ("temperature", "humidity", "pressure"):
            v = observation.get(k)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                all_null = False
                break
        if all_null:
            return (
                True,
                "COMMUNICATION_OVERRIDE",
                "Complete telemetry transmission packet missing all core meteorological parameters",
            )

    # 3. Quality flags inspection
    if any("OUT_OF_RANGE" in f for f in flags):
        return (
            True,
            "DATA_QUALITY_OVERRIDE",
            f"Sensor reading flagged out-of-range: {', '.join(f for f in flags if 'OUT_OF_RANGE' in f)}",
        )

    if any("TIMESTAMP" in f for f in flags):
        return (
            True,
            "TIMESTAMP_OVERRIDE",
            f"Timestamp corruption detected: {', '.join(f for f in flags if 'TIMESTAMP' in f)}",
        )

    return False, None, None


def fuse_evidence(
    evidence: dict[str, float | None],
    weights: dict[str, float],
    quality_flags: list[str] | None = None,
    observation: dict[str, Any] | None = None,
    anomaly_run_length: int = 1,
) -> dict[str, Any]:
    """
    Synthesizes multi-channel evidence with dynamic re-weighting,
    deterministic override precedence, conflict tracking, and traceable provenance.
    """
    # 1. Evaluate deterministic physical/transmission overrides
    is_override, override_type, override_reason = check_deterministic_overrides(
        observation=observation,
        quality_flags=quality_flags,
    )

    # 2. Separate available vs unavailable channels and normalize weights
    available_sources: list[str] = []
    unavailable_sources: list[str] = []
    channel_models: dict[str, ChannelEvidence] = {}

    total_available_weight = 0.0
    for source, value in evidence.items():
        weight = float(weights.get(source, 0.0))
        if value is None:
            unavailable_sources.append(source)
            channel_models[source] = ChannelEvidence(
                channel=source,
                status="UNAVAILABLE",
                score=None,
                weight=0.0,
                weighted_contribution=0.0,
                provenance="EXTERNAL_CONTEXT" if source in ("spatial", "forecast") else "AWS_TELEMETRY",
            )
        else:
            available_sources.append(source)
            total_available_weight += weight

    # 3. Compute weighted sum with dynamic re-weighting over available channels
    weighted_sum = 0.0
    for source in available_sources:
        raw_val = float(evidence[source])  # type: ignore
        clamped_val = max(0.0, min(1.0, raw_val))
        base_weight = float(weights.get(source, 0.0))

        # Dynamic re-normalization of weight
        re_normalized_weight = (
            (base_weight / total_available_weight)
            if total_available_weight > 0.0
            else 0.0
        )
        contribution = clamped_val * re_normalized_weight
        weighted_sum += contribution

        channel_models[source] = ChannelEvidence(
            channel=source,
            status="AVAILABLE",
            score=clamped_val,
            weight=re_normalized_weight,
            weighted_contribution=contribution,
            provenance="EXTERNAL_CONTEXT" if source in ("spatial", "forecast") else "AWS_TELEMETRY",
        )

    # 4. Handle edge case: no valid channels available
    if total_available_weight == 0.0 and not is_override:
        return {
            "score": 0.0,
            "used_sources": [],
            "unavailable_sources": unavailable_sources,
            "status": "INSUFFICIENT_DATA",
            "channels": {k: v.to_dict() for k, v in channel_models.items()},
            "has_conflict": False,
            "conflict_details": None,
            "is_override": False,
            "override_reason": None,
            "persistence": {
                "run_length": anomaly_run_length,
                "is_persistent": False,
                "classification": "TRANSIENT",
            },
        }

    # 5. Evaluate potential conflict between internal telemetry and external context
    # High sensor excursion vs high external normalcy or vice-versa
    sensor_scores = [
        evidence[s] for s in ("ml", "temporal", "statistical", "multivariate")
        if evidence.get(s) is not None
    ]
    context_scores = [
        evidence[s] for s in ("spatial", "forecast")
        if evidence.get(s) is not None
    ]

    has_conflict = False
    conflict_details: str | None = None
    if sensor_scores and context_scores:
        avg_sensor = float(np.mean(sensor_scores))
        avg_context = float(np.mean(context_scores))
        divergence = abs(avg_sensor - avg_context)
        if divergence >= 0.45:
            has_conflict = True
            conflict_details = (
                f"Evidence divergence of {divergence:.2f} detected between sensor telemetry "
                f"({avg_sensor:.2f}) and contextual observations ({avg_context:.2f})"
            )

    # 6. Apply override score if triggered
    if is_override:
        final_score = 1.0
        status = override_type or "DATA_QUALITY_OVERRIDE"
    else:
        final_score = float(weighted_sum)
        if has_conflict:
            status = "CONFLICTING_EVIDENCE"
        else:
            status = "AVAILABLE"

    # 7. Persistence state evaluation
    is_persistent = anomaly_run_length >= 3
    persistence_state = (
        "PERSISTENT_ANOMALY"
        if is_persistent
        else ("REPEATED_ANOMALY" if anomaly_run_length > 1 else "TRANSIENT_ANOMALY")
    )

    return {
        "score": round(final_score, 4),
        "used_sources": available_sources,
        "unavailable_sources": unavailable_sources,
        "status": status,
        "channels": {k: v.to_dict() for k, v in channel_models.items()},
        "has_conflict": has_conflict,
        "conflict_details": conflict_details,
        "is_override": is_override,
        "override_reason": override_reason,
        "persistence": {
            "run_length": anomaly_run_length,
            "is_persistent": is_persistent,
            "classification": persistence_state,
        },
    }
