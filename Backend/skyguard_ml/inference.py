from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .anomaly_types import infer_anomaly_type, infer_affected_parameter
from .config import TrainingConfig
from .evidence_fusion import fuse_evidence
from .event_context import classify_event_context
from .explainability import (
    build_explanation,
    build_reason_codes,
    build_structured_explanation,
    recommended_action,
)

from .feature_engineering import BASE_VARIABLES
from .schemas import PredictionResult, normalize_observation
from .sensor_health import SensorHealthTracker
from .spatial_analysis import analyze_spatial_context
from .forecast_analysis import analyze_forecast_context
from .statistical_detectors import run_statistical_detectors
from .validation import validate_observations
from .lof_adapter import SkyGuardLOFAdapter


class SkyGuard:
    def __init__(
        self,
        artifact_dir: str | Path = "skyguard_ml/artifacts",
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.bundle: dict[str, Any] | None = None
        self.health_tracker = SensorHealthTracker()
        self.lof_adapter = SkyGuardLOFAdapter()

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    def load(self) -> "SkyGuard":
        path = self.artifact_dir / "best_model.joblib"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing trained artifact: {path}"
            )

        self.bundle = joblib.load(path)
        return self

    def train(
        self,
        observations: list[dict[str, Any]] | pd.DataFrame,
        config: TrainingConfig | None = None,
    ) -> dict[str, Any]:
        from .training import train_all

        report = train_all(
            observations,
            config=config,
            artifact_dir=self.artifact_dir,
        )
        self.load()
        return report

    def predict(
        self,
        observation: dict[str, Any],
        historical_context: list[dict[str, Any]] | pd.DataFrame,
        forecast_context: dict[str, Any] | None = None,
        nearby_station_context: list[dict[str, Any]]
        | pd.DataFrame
        | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            self.load()

        assert self.bundle is not None

        current = normalize_observation(observation)

        if isinstance(historical_context, pd.DataFrame):
            history_records = [normalize_observation(r) for r in historical_context.to_dict(orient="records")]
            history = pd.DataFrame(history_records)
        elif historical_context:
            history_records = [normalize_observation(r) for r in historical_context]
            history = pd.DataFrame(history_records)
        else:
            history = pd.DataFrame()

        combined = pd.concat(
            [
                history,
                pd.DataFrame([current]),
            ],
            ignore_index=True,
        )

        validated, validation_report = validate_observations(
            combined,
        )

        validated = validated.sort_values(
            ["station_id", "timestamp"],
            kind="mergesort",
        ).reset_index(drop=True)

        builder = self.bundle["feature_builder"]
        baseline = self.bundle["baseline"]
        model = self.bundle["model"]

        features = builder.transform(validated)

        current_station = current["station_id"]
        current_timestamp = current["timestamp"]

        matching = features[
            (features["station_id"] == current_station)
            & (features["timestamp"] == current_timestamp)
        ]

        if matching.empty:
            matching = features.tail(1)

        current_features = matching.iloc[-1]

        x_current = current_features.reindex(
            builder.feature_list,
        ).to_frame().T

        ml_score = float(
            model.score_samples(x_current)[0]
        )

        lof_result = None
        if hasattr(self, "lof_adapter") and self.lof_adapter and self.lof_adapter.is_ready():
            try:
                lof_result = self.lof_adapter.predict(
                    station_id=str(current_station),
                    observation=current,
                    history=historical_context or [],
                )
                if lof_result.get("status") == "EVALUATED":
                    extracted = self.lof_adapter.extract_evidence_score(lof_result)
                    if extracted is not None and self.bundle.get("model_name") == "LOF":
                        ml_score = extracted
            except Exception:
                lof_result = None

        target_history = validated[
            validated["station_id"] == current_station
        ].copy()

        statistical = run_statistical_detectors(
            history=target_history,
            current_features=current_features,
            variables=BASE_VARIABLES,
            frozen_tolerance=builder.config.frozen_tolerance,
            frozen_duration_hours=builder.config.frozen_duration_hours,
        )

        historical_scores: list[float] = []
        baseline_sources: dict[str, str] = {}
        baseline_details: dict[str, Any] = {}

        for variable in BASE_VARIABLES:
            score, source, stats = baseline.score(
                variable=variable,
                value=current.get(variable),
                station_id=current_station,
                timestamp=current_timestamp,
            )
            historical_scores.append(score)
            baseline_sources[variable] = source
            baseline_details[variable] = stats

        historical_score = max(historical_scores or [0.0])
        temporal_score = float(
            statistical.get("temporal_score", 0.0)
        )

        multivariate_values = [
            abs(float(current_features.get(column, 0.0)))
            for column in [
                "temperature_humidity_deviation",
                "temperature_pressure_deviation",
                "humidity_pressure_deviation",
            ]
            if pd.notna(current_features.get(column))
        ]

        multivariate_score = min(
            1.0,
            (sum(multivariate_values) / len(multivariate_values))
            if multivariate_values
            else 0.0,
        )

        spatial = analyze_spatial_context(
            observation=current,
            nearby_context=nearby_station_context,
            variables=BASE_VARIABLES,
        )

        forecast = analyze_forecast_context(
            observation=current,
            forecast_context=forecast_context,
            variables=BASE_VARIABLES,
        )

        spatial_score = spatial.get("spatial_score")
        forecast_score = forecast.get("forecast_score")

        data_quality_score = 1.0 - validation_report.quality_score

        evidence_values = {
            "data_quality": data_quality_score,
            "statistical": historical_score,
            "temporal": temporal_score,
            "ml": ml_score,
            "multivariate": multivariate_score,
            "spatial": spatial_score,
            "forecast": forecast_score,
        }

        weights = self.bundle["config"].fusion_weights.as_dict()
        threshold = float(self.bundle.get("threshold", 0.40))

        current_quality_flags: list[str] = []
        if "quality_flags" in validated.columns and not validated.empty:
            current_matching = validated[
                (validated["station_id"] == current_station)
                & (validated["timestamp"] == current_timestamp)
            ]
            if not current_matching.empty:
                current_quality_flags = current_matching.iloc[-1].get("quality_flags") or []
            else:
                current_quality_flags = validated.iloc[-1].get("quality_flags") or []

        # Delta calculation for thermodynamic & rate-of-change reasoning
        deltas: dict[str, float | None] = {}
        if not target_history.empty and len(target_history) >= 2:
            prev_obs = target_history.iloc[-2]
            for var in ("temperature", "humidity", "pressure"):
                cur_v = current.get(var)
                prev_v = prev_obs.get(var)
                if pd.notna(cur_v) and pd.notna(prev_v):
                    try:
                        deltas[var] = float(cur_v) - float(prev_v)
                    except (ValueError, TypeError):
                        pass

        fused = fuse_evidence(
            evidence=evidence_values,
            weights=weights,
            quality_flags=current_quality_flags,
            observation=current,
        )

        anomaly_score = float(fused["score"])

        if anomaly_score >= 0.85:
            status = "CRITICAL"
        elif anomaly_score >= 0.65:
            status = "ANOMALY"
        elif anomaly_score >= threshold:
            status = "WARNING"
        else:
            status = "NORMAL"

        is_warmup = len(target_history) < 24

        event_classification = classify_event_context(
            anomaly_score=anomaly_score,
            historical_score=historical_score,
            temporal_score=temporal_score,
            multivariate_score=multivariate_score,
            spatial_score=spatial_score,
            forecast_score=forecast_score,
            observation=current,
            deltas=deltas,
            spatial_status=spatial.get("status"),
            has_conflict=fused.get("has_conflict", False),
        )

        anomaly_type = infer_anomaly_type(
            statistical=statistical,
            quality_flags=current_quality_flags,
            multivariate_score=multivariate_score,
            is_warmup=is_warmup,
            event_context=event_classification,
            fused_score=anomaly_score,
            threshold=threshold,
        )

        affected_param = infer_affected_parameter(
            anomaly_type=anomaly_type,
            statistical=statistical,
            quality_flags=current_quality_flags,
            deltas=deltas,
        )

        severity = self._severity(
            anomaly_score=anomaly_score,
            temporal_score=temporal_score,
            historical_score=historical_score,
            anomaly_type=anomaly_type,
            quality_score=data_quality_score,
        )

        health = self.health_tracker.update(
            station_id=str(current_station),
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            missing_score=statistical["missing"]["score"],
            drift_score=max(
                (
                    value["score"]
                    for value in statistical["drift"].values()
                ),
                default=0.0,
            ),
            spatial_disagreement=spatial.get(
                "sensor_disagreement_evidence"
            ),
            forecast_score=forecast_score,
        )

        evidence = {
            "ml_score": round(ml_score, 4),
            "historical_score": round(historical_score, 4),
            "temporal_score": round(temporal_score, 4),
            "multivariate_score": round(
                multivariate_score,
                4,
            ),
            "spatial_score": (
                None
                if spatial_score is None
                else round(spatial_score, 4)
            ),
            "forecast_score": (
                None
                if forecast_score is None
                else round(forecast_score, 4)
            ),
            "data_quality_score": round(
                data_quality_score,
                4,
            ),
            "baseline_sources": baseline_sources,
            "baseline_statistics": baseline_details,
            "statistical_detectors": statistical,
            "spatial": spatial,
            "forecast": forecast,
            "quality_flags": current_quality_flags,
            "used_sources": fused["used_sources"],
            "unavailable_sources": fused["unavailable_sources"],
            "channels": fused.get("channels", {}),
            "has_conflict": fused.get("has_conflict", False),
            "conflict_details": fused.get("conflict_details"),
            "is_override": fused.get("is_override", False),
            "override_reason": fused.get("override_reason"),
            "persistence": fused.get("persistence", {}),
            "lof": lof_result,
        }

        reason_codes = build_reason_codes(
            anomaly_type=anomaly_type,
            historical_score=historical_score,
            temporal_score=temporal_score,
            multivariate_score=multivariate_score,
            spatial_score=spatial_score,
            forecast_score=forecast_score,
            quality_flags=current_quality_flags,
            ml_score=ml_score,
            threshold=threshold,
        )

        structured_explanation = build_structured_explanation(
            status=status,
            anomaly_type=anomaly_type,
            evidence=evidence,
            reason_codes=reason_codes,
            event_classification=event_classification,
            observation=current,
            deltas=deltas,
            affected_parameter=affected_param,
            threshold=threshold,
        )

        explanation = build_explanation(
            status=status,
            anomaly_type=anomaly_type,
            evidence=evidence,
            reason_codes=reason_codes,
            event_classification=event_classification,
            observation=current,
            deltas=deltas,
            affected_parameter=affected_param,
        )

        action = recommended_action(
            status=status,
            event_classification=event_classification,
            anomaly_type=anomaly_type,
            affected_parameter=affected_param,
        )

        waterfall = [
            {
                "stage": "OBSERVATION",
                "status": "PASS",
                "summary": f"Station {current_station} telemetry ingested at {current_timestamp}",
            },
            {
                "stage": "DATA_QUALITY",
                "status": "FAIL" if fused.get("is_override") else ("WARNING" if data_quality_score > 0.10 else "PASS"),
                "score": round(data_quality_score, 4),
                "summary": fused.get("override_reason") or ("Quality checks passed" if data_quality_score <= 0.10 else f"Quality penalty: {data_quality_score:.2f}"),
            },
            {
                "stage": "TEMPORAL",
                "status": "FAIL" if temporal_score >= 0.70 else ("WARNING" if temporal_score >= threshold else "PASS"),
                "score": round(temporal_score, 4),
                "summary": f"Rate of change score {temporal_score:.2f} (threshold: {threshold:.2f})",
            },
            {
                "stage": "STATISTICAL",
                "status": "FAIL" if historical_score >= 0.70 else ("WARNING" if historical_score >= threshold else "PASS"),
                "score": round(historical_score, 4),
                "summary": f"Historical baseline deviation {historical_score:.2f}",
            },
            {
                "stage": "ML_DETECTOR",
                "status": "FAIL" if ml_score >= 0.70 else ("WARNING" if ml_score >= threshold else "PASS"),
                "score": round(ml_score, 4),
                "threshold": threshold,
                "summary": f"LOF score {ml_score:.2f} evaluated against decision threshold {threshold:.2f}",
            },
            {
                "stage": "MULTIVARIATE",
                "status": "FAIL" if multivariate_score >= 0.70 else ("WARNING" if multivariate_score >= threshold else "PASS"),
                "score": round(multivariate_score, 4),
                "summary": f"Inter-parameter covariance deviation {multivariate_score:.2f}",
            },
            {
                "stage": "SPATIAL_CONTEXT",
                "status": "SKIPPED" if spatial_score is None else ("PASS" if spatial.get("status") == "SPATIAL_CORROBORATED" else "WARNING"),
                "score": round(spatial_score, 4) if spatial_score is not None else None,
                "summary": f"Spatial context: {spatial.get('status', 'SPATIAL_EVIDENCE_UNAVAILABLE')}",
            },
            {
                "stage": "EVIDENCE_FUSION",
                "status": "OVERRIDE" if fused.get("is_override") else ("WARNING" if anomaly_score >= threshold else "PASS"),
                "score": round(anomaly_score, 4),
                "threshold": threshold,
                "summary": f"Fused score {anomaly_score:.2f} across {len(fused.get('used_sources', []))} active channels",
            },
            {
                "stage": "CLASSIFICATION",
                "status": "WARNING" if status != "NORMAL" else "PASS",
                "summary": f"Classified as {status} [{anomaly_type}] on parameter: {affected_param}",
            },
        ]

        result = PredictionResult(
            status=status,
            anomaly_score=anomaly_score,
            selected_model=self.bundle["model_name"],
            anomaly_type=anomaly_type,
            severity=severity,
            event_classification=event_classification,
            evidence=evidence,
            sensor_health=health,
            reason_codes=reason_codes,
            explanation=explanation,
            recommended_action=action,
            confidence_status="NOT_CALIBRATED",
            affected_parameter=affected_param,
            structured_explanation=structured_explanation,
            evidence_waterfall=waterfall,
        )

        return result.to_dict()


    @staticmethod
    def _severity(
        anomaly_score: float,
        temporal_score: float,
        historical_score: float,
        anomaly_type: str,
        quality_score: float,
    ) -> str:
        score = (
            0.40 * anomaly_score
            + 0.20 * temporal_score
            + 0.20 * historical_score
            + 0.20 * quality_score
        )

        if anomaly_type in {
            "COMMUNICATION_FAILURE",
            "MISSING_DATA",
            "TIMESTAMP_ERROR",
        }:
            score += 0.10

        if score >= 0.85:
            return "CRITICAL"
        if score >= 0.65:
            return "HIGH"
        if score >= 0.40:
            return "MEDIUM"
        return "LOW"
