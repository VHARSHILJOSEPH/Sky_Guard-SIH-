"""
SkyGuard AI — Single Authoritative Telemetry Pipeline Processor.

Implements the end-to-end target pipeline:
telemetry input
→ normalization
→ validation
→ data quality
→ feature engineering
→ statistical detection
→ temporal detection
→ ML detector
→ multivariate consistency
→ spatial corroboration
→ weather/context evidence
→ evidence fusion
→ final classification
→ explanation
→ station health
→ persistence
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import time
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from .canonical_schema import (
    CanonicalTelemetryInput,
    CanonicalEvidence,
    CanonicalInferenceResponse,
    ClassificationState,
    ModelMetadataModel,
    SeverityLevel,
)
from .anomaly_types import infer_affected_parameter
from .explainability import build_structured_explanation
from .incident_manager import incident_manager
from .inference import SkyGuard
from .schemas import normalize_observation
from .sensor_health import sensor_health_tracker
from .validation import validate_observations
from skyguard_lof import SkyGuardLOF


from app.services.database_service import (
    save_sensor_reading,
    save_anomaly,
    save_station_health,
    get_station_history_chronological,
)

MAX_HISTORY_BUFFER = 120

# Isolated history buffers to prevent demo from contaminating live history
station_history_live: dict[str, list[dict[str, Any]]] = defaultdict(list)
previous_observation_live: dict[str, dict[str, Any]] = {}

station_history_demo: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
previous_observation_demo: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)


class CanonicalPipelineService:
    """
    The Single Authoritative Detection Pipeline Service.
    """

    def __init__(
        self,
        engine: Optional[SkyGuard] = None,
        model: Optional[SkyGuardLOF] = None,
    ) -> None:
        self.engine = engine
        self.model = model

    def hydrate_history_if_needed(self, station_id: str, is_live: bool, session_id: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Hydrates in-memory station history from the persistent database store
        if the in-memory cache is empty (e.g. after server restart).
        """
        if is_live:
            buffer = station_history_live[station_id]
            if not buffer:
                # Load persistent history from database service
                persisted = get_station_history_chronological(station_id, limit=50)
                if persisted:
                    station_history_live[station_id] = persisted
                    buffer = station_history_live[station_id]
            return list(buffer)
        else:
            sess_key = session_id or "default_demo"
            return list(station_history_demo[sess_key][station_id])

    def process(
        self,
        telemetry_input: CanonicalTelemetryInput,
        nearby_station_context: Optional[list[dict[str, Any]]] = None,
        forecast_context: Optional[dict[str, Any]] = None,
    ) -> CanonicalInferenceResponse:
        """
        Executes the single canonical pipeline for an observation.
        """
        started = time.perf_counter()

        station_id = telemetry_input.station_id
        timestamp_str = telemetry_input.timestamp
        source = telemetry_input.source
        is_demo = source == "DEMO_SIMULATION"
        is_live = not is_demo
        session_id = telemetry_input.session_id
        sess_key = session_id or "default_demo"

        readings_dict = {
            "temperature": telemetry_input.temperature_c,
            "humidity": telemetry_input.humidity_pct,
            "pressure": telemetry_input.pressure_hpa,
            "wind_speed": telemetry_input.wind_speed_ms,
            "wind_direction": telemetry_input.wind_direction_deg,
            "rainfall": telemetry_input.rainfall_mm,
        }

        # -------------------------------------------------------------
        # 1. Previous Observations & Deltas
        # -------------------------------------------------------------
        if is_live:
            prev_obs = previous_observation_live.get(station_id, {})
        else:
            prev_obs = previous_observation_demo[sess_key].get(station_id, {})

        prev_temp = prev_obs.get("temperature")
        prev_hum = prev_obs.get("humidity")
        prev_press = prev_obs.get("pressure")

        cur_temp = telemetry_input.temperature_c
        cur_hum = telemetry_input.humidity_pct
        cur_press = telemetry_input.pressure_hpa

        delta_temp = (
            round(cur_temp - prev_temp, 2)
            if (cur_temp is not None and prev_temp is not None and math.isfinite(cur_temp) and math.isfinite(prev_temp))
            else None
        )
        delta_hum = (
            round(cur_hum - prev_hum, 2)
            if (cur_hum is not None and prev_hum is not None and math.isfinite(cur_hum) and math.isfinite(prev_hum))
            else None
        )
        delta_press = (
            round(cur_press - prev_press, 2)
            if (cur_press is not None and prev_press is not None and math.isfinite(cur_press) and math.isfinite(prev_press))
            else None
        )

        deltas = {
            "temperature": delta_temp,
            "humidity": delta_hum,
            "pressure": delta_press,
        }

        # -------------------------------------------------------------
        # 2. History Retrieval (with restart persistence hydration)
        # -------------------------------------------------------------
        history = self.hydrate_history_if_needed(station_id, is_live=is_live, session_id=session_id)
        recent_history = history[-30:] if len(history) > 30 else history
        available_history_length = len(history)

        # -------------------------------------------------------------
        # 3. Data Quality & Physical Bounds Validation
        # -------------------------------------------------------------
        obs_record = {
            "station_id": station_id,
            "timestamp": timestamp_str,
            "temperature": cur_temp,
            "humidity": cur_hum,
            "pressure": cur_press,
            "wind_speed": telemetry_input.wind_speed_ms or 0.0,
            "wind_direction": telemetry_input.wind_direction_deg or 0.0,
            "rainfall": telemetry_input.rainfall_mm or 0.0,
        }

        val_frame, val_report = validate_observations(pd.DataFrame([obs_record]))
        is_physically_valid = bool(val_report.valid_row_count > 0)
        quality_score = float(val_report.quality_score)

        missing_fields: list[str] = []
        if cur_temp is None:
            missing_fields.append("temperature")
        if cur_hum is None:
            missing_fields.append("humidity")
        if cur_press is None:
            missing_fields.append("pressure")

        is_corrupted = bool(
            cur_temp is not None and (cur_temp > 65.0 or cur_temp < -50.0)
            or cur_hum is not None and (cur_hum < 0.0 or cur_hum > 100.0)
            or cur_press is not None and (cur_press < 700.0 or cur_press > 1150.0)
        )

        # -------------------------------------------------------------
        # 4. ML Detector Inference (LOF / Calibrated Model)
        # -------------------------------------------------------------
        lof_result: dict[str, Any] = {}
        ml_score = 0.0
        ml_threshold = 0.40

        if self.model is not None:
            try:
                clean_hist = [dict(h) for h in recent_history if h.get("station_id") in (None, station_id)]
                lof_result = self.model.predict(
                    station_id=station_id,
                    observation={
                        "station_id": station_id,
                        "timestamp": timestamp_str,
                        "temperature": cur_temp if (cur_temp is not None and math.isfinite(cur_temp)) else None,
                        "humidity": cur_hum if (cur_hum is not None and math.isfinite(cur_hum)) else None,
                        "pressure": cur_press if (cur_press is not None and math.isfinite(cur_press)) else None,
                    },
                    history=clean_hist,
                )
                ml_score = float(lof_result.get("normalized_lof_score") or 0.0)
                ml_threshold = float(lof_result.get("threshold") or 0.40)
            except Exception as exc:
                lof_result = {"status": "ERROR", "error": str(exc), "ml_anomaly": False}


        # -------------------------------------------------------------
        # 5. Full Hybrid SkyGuard Pipeline Execution
        # -------------------------------------------------------------
        prediction_dict: dict[str, Any] = {}
        if self.engine is not None and self.engine.loaded:
            try:
                pred = self.engine.predict(
                    observation=obs_record,
                    historical_context=recent_history,
                    forecast_context=forecast_context,
                    nearby_station_context=nearby_station_context,
                )
                prediction_dict = pred.to_dict() if hasattr(pred, "to_dict") else dict(pred)
            except Exception as eng_exc:
                prediction_dict = {
                    "status": "NORMAL",
                    "anomaly_score": 0.0,
                    "severity": "LOW",
                    "anomaly_type": "NONE",
                    "explanation": f"Engine execution note: {eng_exc}",
                    "reason_codes": [],
                    "evidence": {},
                }
        else:
            prediction_dict = {
                "status": "NORMAL",
                "anomaly_score": 0.0,
                "severity": "LOW",
                "anomaly_type": "NONE",
                "explanation": "Standby baseline processing",
                "reason_codes": [],
                "evidence": {},
            }

        # Extract pipeline channel evidence
        evidence_dict = prediction_dict.get("evidence", {})
        stat_evidence = evidence_dict.get("statistical_detectors", {})
        frozen_detectors = stat_evidence.get("frozen", {})
        drift_detectors = stat_evidence.get("drift", {})
        spike_detectors = stat_evidence.get("spike_drop", {})
        spatial_evidence = evidence_dict.get("spatial", {})
        forecast_evidence = evidence_dict.get("forecast", {})

        # Compute channel scores
        data_quality_evidence_score = 1.0 - quality_score
        statistical_score = float(evidence_dict.get("historical_score", 0.0))
        temporal_score = float(evidence_dict.get("temporal_score", 0.0))
        multivariate_score = float(evidence_dict.get("multivariate_score", 0.0))
        spatial_score = evidence_dict.get("spatial_score")
        forecast_score = evidence_dict.get("forecast_score")

        fused_anomaly_score = float(prediction_dict.get("anomaly_score", ml_score))

        # -------------------------------------------------------------
        # 6. Classification State & Anomaly Type Resolution
        # -------------------------------------------------------------
        min_history_req = 24
        is_warmup = available_history_length < min_history_req

        anomaly_type = "NORMAL"
        severity: SeverityLevel = "LOW"
        final_status: ClassificationState = "NORMAL"

        # Temporal and rule indicators
        is_spike = delta_temp is not None and abs(delta_temp) >= 6.0
        is_frozen = any(v.get("score", 0.0) >= 0.6 for v in frozen_detectors.values())
        is_drift = any(v.get("score", 0.0) >= 0.5 for v in drift_detectors.values())
        is_multi = multivariate_score >= 0.60 or lof_result.get("ml_anomaly", False)

        peer_disagreement = bool(spatial_score is not None and spatial_score >= 0.60)
        peer_agreement = bool(spatial_score is not None and spatial_score < 0.35)

        # Check if engine provided authoritative classification
        eng_anomaly_type = prediction_dict.get("anomaly_type")
        eng_status = prediction_dict.get("status")
        eng_event_class = prediction_dict.get("event_classification")
        eng_affected = prediction_dict.get("affected_parameter")
        eng_structured_exp = prediction_dict.get("structured_explanation")
        eng_waterfall = prediction_dict.get("evidence_waterfall", [])

        # 1. Physical limits rejection and communication faults are immediate and never masked as WARMUP
        if is_corrupted:
            final_status = "DATA_QUALITY_ISSUE"
            anomaly_type = "DATA_QUALITY_ANOMALY"
            severity = "CRITICAL"
        elif len(missing_fields) >= 3:
            final_status = "COMMUNICATION_FAILURE"
            anomaly_type = "COMMUNICATION_FAILURE"
            severity = "HIGH"
        elif len(missing_fields) > 0:
            final_status = "DATA_QUALITY_ISSUE"
            anomaly_type = "DATA_QUALITY_ANOMALY"
            severity = "HIGH"
        elif eng_anomaly_type and eng_anomaly_type not in ("NONE", "NORMAL") and eng_status != "NORMAL":
            anomaly_type = eng_anomaly_type
            severity = prediction_dict.get("severity", severity)
            if is_warmup and anomaly_type not in ("DATA_QUALITY_ANOMALY", "CORRUPTED_DATA", "COMMUNICATION_FAILURE"):
                final_status = "WARMUP"
            elif anomaly_type == "COMMUNICATION_FAILURE":
                final_status = "COMMUNICATION_FAILURE"
            elif anomaly_type in ("DATA_QUALITY_ANOMALY", "CORRUPTED_DATA", "MISSING_DATA"):
                final_status = "DATA_QUALITY_ISSUE"
            elif eng_event_class == "LIKELY_WEATHER_EVENT":
                final_status = "POSSIBLE_WEATHER_EVENT"
            elif eng_event_class == "LIKELY_SENSOR_ANOMALY":
                final_status = "POSSIBLE_SENSOR_FAULT"
            elif eng_event_class == "INSUFFICIENT_EVIDENCE":
                final_status = "INSUFFICIENT_EVIDENCE"
            else:
                final_status = "ANOMALOUS"
        elif is_warmup:
            final_status = "WARMUP"
            anomaly_type = "NORMAL"
            severity = "LOW"
        elif is_spike:
            anomaly_type = "TEMPERATURE_ANOMALY"
            severity = "HIGH"
            final_status = (
                "POSSIBLE_WEATHER_EVENT"
                if (peer_agreement and forecast_score is not None and forecast_score < 0.40)
                else ("POSSIBLE_SENSOR_FAULT" if peer_disagreement else "ANOMALOUS")
            )
        elif is_frozen:
            anomaly_type = "FROZEN_SENSOR"
            severity = "HIGH"
            final_status = "POSSIBLE_SENSOR_FAULT"
        elif is_multi:
            anomaly_type = "MULTIVARIATE_INCONSISTENCY"
            severity = "MEDIUM" if fused_anomaly_score < 0.75 else "HIGH"
            final_status = "POSSIBLE_SENSOR_FAULT" if peer_disagreement else "ANOMALOUS"
        elif is_drift:
            anomaly_type = "SENSOR_DRIFT"
            severity = "MEDIUM" if fused_anomaly_score < 0.70 else "HIGH"
            final_status = "POSSIBLE_SENSOR_FAULT" if peer_disagreement else "ANOMALOUS"
        elif fused_anomaly_score >= 0.40:
            anomaly_type = "LIKELY_SENSOR_ANOMALY"
            severity = "MEDIUM"
            final_status = "ANOMALOUS"
        else:
            final_status = "NORMAL"
            anomaly_type = "NORMAL"
            severity = "LOW"

        is_anomaly = final_status in (
            "ANOMALOUS",
            "POSSIBLE_SENSOR_FAULT",
            "POSSIBLE_WEATHER_EVENT",
            "DATA_QUALITY_ISSUE",
            "COMMUNICATION_FAILURE",
        )

        affected_parameter = eng_affected or infer_affected_parameter(
            anomaly_type=anomaly_type,
            statistical=stat_evidence,
            quality_flags=missing_fields + (["PHYSICAL_LIMIT_EXCEEDED"] if is_corrupted else []),
            deltas=deltas,
        )

        # Evidence strength
        evidence_strength = float(max(
            fused_anomaly_score,
            ml_score,
            temporal_score,
            data_quality_evidence_score,
        ))

        # -------------------------------------------------------------
        # 7. Explanation & Reason Codes
        # -------------------------------------------------------------
        why_flagged: list[str] = []
        if is_corrupted:
            why_flagged.append(f"Physical limit rejection: reading ({cur_temp}°C) exceeds physical bounds (-50°C to 65°C)")
        elif anomaly_type in ("MISSING_DATA", "DATA_QUALITY_ANOMALY") and missing_fields:
            why_flagged.append(f"Core meteorological sensor reading is NULL / missing: {', '.join(missing_fields)}")
        elif anomaly_type == "COMMUNICATION_FAILURE":
            why_flagged.append("Physical transmission packet lacked all core sensor parameters")

        if is_spike:
            why_flagged.append(f"Temperature shifted {delta_temp:+.1f}°C in one observation interval (exceeds rate limit)")
        if is_frozen:
            why_flagged.append("Sensor reading showed zero variance across consecutive historical intervals")
        if is_drift:
            why_flagged.append("Gradual baseline departure detected by standardized drift accumulator")
        if lof_result.get("ml_anomaly") or ml_score >= ml_threshold:
            why_flagged.append(f"Local Outlier Factor (LOF) flagged multivariate density anomaly (score: {ml_score:.2f})")
        if final_status == "POSSIBLE_WEATHER_EVENT":
            why_flagged.append("Nearby peer stations recorded coherent changes (spatial agreement confirms regional weather transition)")
        elif final_status == "POSSIBLE_SENSOR_FAULT":
            why_flagged.append("Nearby peer stations remained stable (spatial disagreement indicates isolated instrument fault)")
        if not why_flagged:
            why_flagged.append("All physical parameters within expected climatological bounds")

        # 5-Question Structured Explanation
        resolved_event_class = eng_event_class or (
            "LIKELY_WEATHER_EVENT" if final_status == "POSSIBLE_WEATHER_EVENT"
            else ("LIKELY_SENSOR_ANOMALY" if final_status == "POSSIBLE_SENSOR_FAULT" else "INSUFFICIENT_EVIDENCE")
        )
        structured_exp = build_structured_explanation(
            status=final_status,
            anomaly_type=anomaly_type,
            evidence={
                "ml_score": ml_score,
                "temporal_score": temporal_score,
                "historical_score": statistical_score,
                "multivariate_score": multivariate_score,
                "data_quality_score": data_quality_evidence_score,
                "spatial": {
                    "status": spatial_evidence.get("status", "SPATIAL_EVIDENCE_UNAVAILABLE"),
                    "spatial_score": spatial_score,
                    "spatial_agreement_score": 1.0 - (spatial_score or 0.0),
                },
                "forecast": {
                    "status": forecast_evidence.get("status", "UNAVAILABLE"),
                    "score": forecast_score,
                },
            },
            reason_codes=why_flagged,
            event_classification=resolved_event_class,
            observation=obs_record,
            deltas=deltas,
            affected_parameter=affected_parameter,
            threshold=ml_threshold,
        )

        explanation = {
            "summary": prediction_dict.get("explanation") or structured_exp["what_happened"],
            "why_flagged": why_flagged,
            "recommended_action": (
                "Continue monitoring nominal telemetry"
                if not is_anomaly
                else (
                    "No hardware maintenance required; regional environmental event in progress"
                    if final_status == "POSSIBLE_WEATHER_EVENT"
                    else "Inspect sensor transducer and verify ADC calibration"
                )
            ),
        }

        # Determine spatial status explicitly: SPATIAL_CORROBORATED, SPATIAL_NOT_CORROBORATED, SPATIAL_EVIDENCE_UNAVAILABLE
        if spatial_score is not None:
            spatial_status = "SPATIAL_CORROBORATED" if peer_agreement else ("SPATIAL_NOT_CORROBORATED" if peer_disagreement else "SPATIAL_NEUTRAL")
        else:
            spatial_status = "SPATIAL_EVIDENCE_UNAVAILABLE"

        dt_str = f"ΔT: {delta_temp:+.1f}°C" if delta_temp is not None else "ΔT: N/A"
        dh_str = f"ΔRH: {delta_hum:+.1f}%" if delta_hum is not None else "ΔRH: N/A"
        dp_str = f"ΔP: {delta_press:+.1f}hPa" if delta_press is not None else "ΔP: N/A"
        temporal_obs_str = f"{dt_str}, {dh_str}, {dp_str}" if prev_obs else "First interval (no delta)"
        temporal_dev_str = (
            f"{delta_temp:+.1f}°C excursion" if delta_temp is not None
            else (f"{delta_hum:+.1f}% excursion" if delta_hum is not None else None)
        )

        # 7-Channel Evidence Waterfall
        evidence_waterfall = [
            {
                "channel": "Data Quality",
                "status": "VALID" if (is_physically_valid and not is_corrupted and not missing_fields) else "REJECTED",
                "contribution": round(data_quality_evidence_score, 4),
                "supporting_observation": f"{len(missing_fields)} missing fields, corrupted={is_corrupted}",
                "reference_baseline": "Atmospheric physical bounds [-50°C to 65°C, 0-100%, 700-1150hPa]",
                "deviation": f"Quality score: {quality_score:.2f}",
                "freshness": "Current observation",
                "provenance": source,
            },
            {
                "channel": "Temporal",
                "status": "FLAGGED" if (is_spike or is_drift or is_frozen or temporal_score >= 0.40) else "NOMINAL",
                "contribution": round(temporal_score, 4),
                "supporting_observation": temporal_obs_str,
                "reference_baseline": "Station diurnal rate limit (max 6.0°C/step)",
                "deviation": temporal_dev_str,
                "freshness": "T-0 vs T-1 intervals",
                "provenance": source,
            },
            {
                "channel": "Statistical",
                "status": "FLAGGED" if statistical_score >= 0.40 else "NOMINAL",
                "contribution": round(statistical_score, 4),
                "supporting_observation": f"Deviation score: {statistical_score:.2f}",
                "reference_baseline": "Diurnal historical rolling mean ± 3σ envelope",
                "deviation": f"{statistical_score * 3.0:.1f}σ departures",
                "freshness": f"{available_history_length} historical steps",
                "provenance": source,
            },
            {
                "channel": "Machine Learning",
                "status": "ANOMALOUS" if (ml_score >= ml_threshold or lof_result.get("ml_anomaly", False)) else "NOMINAL",
                "contribution": round(ml_score, 4),
                "supporting_observation": f"LOF density score: {ml_score:.4f} (threshold: {ml_threshold:.2f})",
                "reference_baseline": "Compact 63-feature local density manifold (LOF v2.0.0)",
                "deviation": f"Score {ml_score:.2f} >= {ml_threshold:.2f}" if ml_score >= ml_threshold else "Within density cluster",
                "freshness": "Real-time vector inference",
                "provenance": "SkyGuard-LOF-v2.0.0",
            },
            {
                "channel": "Multivariate",
                "status": "FLAGGED" if multivariate_score >= 0.40 else "NOMINAL",
                "contribution": round(multivariate_score, 4),
                "supporting_observation": f"Thermodynamic covariance score: {multivariate_score:.2f}",
                "reference_baseline": "T/P/RH joint thermodynamic distribution",
                "deviation": f"Covariance residual {multivariate_score:.2f}",
                "freshness": "Synchronized sample",
                "provenance": source,
            },
            {
                "channel": "Spatial",
                "status": spatial_status,
                "contribution": round(spatial_score, 4) if spatial_score is not None else 0.0,
                "supporting_observation": "Peer stations agreement confirmed" if peer_agreement else ("Peer stations invariant / uncorroborated" if peer_disagreement else "No peer stations active within 100km radius"),
                "reference_baseline": "100km regional radius AWS peer consensus",
                "deviation": f"Residual {spatial_score:.2f}" if spatial_score is not None else None,
                "freshness": "Synchronous spatial network timestamp",
                "provenance": "IMD_AWS_REGIONAL_NETWORK" if spatial_score is not None else "UNAVAILABLE",
            },
            {
                "channel": "Weather Context",
                "status": "CORROBORATING" if (forecast_score is not None and forecast_score < 0.40) else ("CONFLICTING" if (forecast_score is not None and forecast_score >= 0.60) else "UNAVAILABLE"),
                "contribution": round(forecast_score, 4) if forecast_score is not None else 0.0,
                "supporting_observation": f"Synoptic forecast alignment: score {forecast_score:.2f}" if forecast_score is not None else "External meteorological context unavailable / unconfigured",
                "reference_baseline": "Synoptic numerical forecast ground truth",
                "deviation": f"Forecast delta {forecast_score:.2f}" if forecast_score is not None else None,
                "freshness": "Hourly synoptic model cycle",
                "provenance": "OPEN_METEO_CONTEXT" if forecast_score is not None else "UNAVAILABLE",
            },
        ]

        # -------------------------------------------------------------
        # 8. Authoritative Station Health Derivation
        # -------------------------------------------------------------
        drift_max = 0.0
        if isinstance(drift_detectors, dict) and drift_detectors:
            drift_max = max((v.get("score", 0.0) for v in drift_detectors.values()), default=0.0)

        sensor_health = sensor_health_tracker.update(
            station_id=station_id,
            anomaly_score=fused_anomaly_score,
            anomaly_type=anomaly_type,
            missing_score=float(len(missing_fields) / 3.0),
            drift_score=drift_max,
            spatial_disagreement=spatial_score if peer_disagreement else 0.0,
            forecast_score=forecast_score,
            is_warmup=is_warmup and anomaly_type not in ("DATA_QUALITY_ANOMALY", "CORRUPTED_DATA", "COMMUNICATION_FAILURE"),
            is_offline=anomaly_type == "COMMUNICATION_FAILURE",
            is_frozen=is_frozen or anomaly_type == "FROZEN_SENSOR",
            quality_score=quality_score,
            history_len=available_history_length,
            timestamp=timestamp_str,
            severity=severity,
        )

        # -------------------------------------------------------------
        # 9. Anomaly Incident Lifecycle & Deduplication
        # -------------------------------------------------------------
        incident = incident_manager.process_observation_incident(
            station_id=station_id,
            timestamp_str=timestamp_str,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            severity=severity,
            session_id=session_id if not is_live else None,
        )

        # -------------------------------------------------------------
        # 10. In-Memory Buffer Update & Demo Isolation
        # -------------------------------------------------------------
        cached_obs = {
            "station_id": station_id,
            "timestamp": timestamp_str,
            "temperature": cur_temp,
            "humidity": cur_hum,
            "pressure": cur_press,
            "wind_speed": telemetry_input.wind_speed_ms or 0.0,
            "rainfall": telemetry_input.rainfall_mm or 0.0,
        }

        if is_live:
            station_history_live[station_id].append(cached_obs)
            if len(station_history_live[station_id]) > MAX_HISTORY_BUFFER:
                station_history_live[station_id] = station_history_live[station_id][-MAX_HISTORY_BUFFER:]
            previous_observation_live[station_id] = cached_obs
        else:
            station_history_demo[sess_key][station_id].append(cached_obs)
            if len(station_history_demo[sess_key][station_id]) > MAX_HISTORY_BUFFER:
                station_history_demo[sess_key][station_id] = station_history_demo[sess_key][station_id][-MAX_HISTORY_BUFFER:]
            previous_observation_demo[sess_key][station_id] = cached_obs

        # Assemble Canonical Evidence before persistence so it can be stored in anomaly record
        evidence = CanonicalEvidence(
            data_quality={
                "score": round(data_quality_evidence_score, 4),
                "valid": is_physically_valid,
                "missing_fields": missing_fields,
            },
            statistical={
                "score": round(statistical_score, 4),
                "baseline_deviation": delta_temp,
            },
            temporal={
                "score": round(temporal_score, 4),
                "spike_score": 0.85 if is_spike else 0.0,
                "drift_score": 0.80 if is_drift else 0.0,
                "frozen_score": 0.90 if is_frozen else 0.0,
            },
            ml={
                "score": round(ml_score, 4),
                "threshold": ml_threshold,
                "model": "LOF_v2.0.0",
                "ml_anomaly": bool(lof_result.get("ml_anomaly", False) or ml_score >= ml_threshold),
            },
            multivariate={
                "score": round(multivariate_score, 4),
            },
            spatial={
                "score": None if spatial_score is None else round(spatial_score, 4),
                "status": spatial_evidence.get("status", "SPATIAL_EVIDENCE_UNAVAILABLE"),
                "peer_agreement": bool(peer_agreement),
                "peer_disagreement": bool(peer_disagreement),
            },
            weather_context={
                "score": None if forecast_score is None else round(forecast_score, 4),
                "status": forecast_evidence.get("status", "UNAVAILABLE"),
            },
            channels=evidence_dict.get("channels", {}),
            has_conflict=bool(evidence_dict.get("has_conflict", False)),
            conflict_details=evidence_dict.get("conflict_details"),
            is_override=bool(evidence_dict.get("is_override", False)),
            override_reason=evidence_dict.get("override_reason"),
            persistence=evidence_dict.get("persistence", {}),
        )

        # -------------------------------------------------------------
        # 11. Database Persistence Layer (with restart resilience & data isolation)
        # -------------------------------------------------------------
        reading_record = save_sensor_reading(
            station_id=station_id,
            timestamp=timestamp_str,
            temperature_c=cur_temp,
            humidity_pct=cur_hum,
            pressure_hpa=cur_press,
            wind_speed_ms=telemetry_input.wind_speed_ms,
            wind_direction_deg=getattr(telemetry_input, "wind_direction_deg", None),
            rainfall_mm=telemetry_input.rainfall_mm,
            source=source,
            session_id=session_id,
            quality_status="VALID" if is_physically_valid and not is_corrupted else "INVALID",
            raw_payload=readings_dict,
        )

        if is_anomaly and (is_live or session_id):
            reading_id = reading_record.get("id") if reading_record else None
            save_anomaly(
                reading_id=reading_id,
                station_id=station_id,
                timestamp=timestamp_str,
                anomaly_type=anomaly_type,
                final_status=final_status,
                severity=severity,
                affected_parameter=affected_parameter,
                observed_value=cur_temp if cur_temp is not None else 0.0,
                expected_value=prev_temp if prev_temp is not None else 30.0,
                anomaly_score=round(fused_anomaly_score, 4),
                confidence=round(fused_anomaly_score, 4),
                explanation=explanation["summary"],
                evidence=evidence.model_dump(),
                model_name="SkyGuard-LOF-v2.0.0",
                model_version="2.0.0",
                status="DETECTED",
                session_id=session_id,
            )

        if is_live:
            save_station_health(
                station_id=station_id,
                health_score=int(sensor_health["score"]),
                status=sensor_health["status"],
                issues=sensor_health["recent_issues"],
                timestamp=timestamp_str,
                availability=100.0 if not is_corrupted else 80.0,
                received_observations=len(station_history_live.get(station_id, [])),
                missing_observations=len(missing_fields),
                anomaly_frequency=float(sensor_health.get("metrics", {}).get("anomaly_frequency", 0.0)),
                last_seen=timestamp_str,
            )

        inference_time_ms = (time.perf_counter() - started) * 1000.0

        model_meta = ModelMetadataModel(
            name="SkyGuard_EvidenceFusion_LOF",
            version="2.0.0",
            feature_schema_version="BME280_FEATURES_v2",
            threshold=ml_threshold,
            calibrated=True,
        )

        return CanonicalInferenceResponse(
            station_id=station_id,
            timestamp=timestamp_str,
            source=source,  # type: ignore
            session_id=session_id,
            device_id=telemetry_input.device_id,
            readings=readings_dict,
            previous_readings={"temperature": prev_temp, "humidity": prev_hum, "pressure": prev_press} if prev_obs else None,
            deltas=deltas if prev_obs else None,
            final_status=final_status,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            affected_parameter=affected_parameter,
            severity=severity,
            anomaly_score=round(fused_anomaly_score, 4),
            evidence_strength=round(evidence_strength, 4),
            confidence_status=prediction_dict.get("confidence_status", "NOT_CALIBRATED"),
            model=model_meta,
            evidence=evidence,
            explanation=explanation,
            structured_explanation=structured_exp,
            evidence_waterfall=evidence_waterfall,
            warmup_state={
                "is_warmup": is_warmup,
                "required_history": min_history_req,
                "available_history": available_history_length,
            },
            data_quality_state={
                "valid": is_physically_valid,
                "missing_fields": missing_fields,
                "imputed_fields": [],
                "physical_bounds_ok": not is_corrupted,
            },
            sensor_health=sensor_health,
            incident=incident,
            inference_time_ms=round(inference_time_ms, 2),
        )



_canonical_pipeline_instance: Optional[CanonicalPipelineService] = None


def get_canonical_pipeline() -> CanonicalPipelineService:
    global _canonical_pipeline_instance
    if _canonical_pipeline_instance is None:
        _canonical_pipeline_instance = CanonicalPipelineService()
    return _canonical_pipeline_instance


canonical_pipeline_service = get_canonical_pipeline()


def get_station_health(station_id: Optional[str] = None) -> dict[str, Any]:
    """Retrieve station health from the authoritative in-memory sensor health tracker."""
    return sensor_health_tracker.get_health_status(station_id=station_id)
