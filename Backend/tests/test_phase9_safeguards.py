"""
SkyGuard AI — Phase 9 Scientific Safeguards Test Suite.

Verifies:
1. Unavailable spatial context is strictly neutral (no penalty, no fake corroboration).
2. Stale spatial peer telemetry (> 3 hours) is excluded.
3. Stale external forecast context (> 3 hours) is excluded.
4. Conflicting evidence is preserved and results in INSUFFICIENT_EVIDENCE.
5. Deterministic physical bounds/communication overrides take absolute precedence.
6. Single vs persistent anomalies are tracked and distinguished.
7. Psychrometric thermodynamic coupling vs isolated sensor fault reasoning.
8. Machine-readable evidence waterfall, 5-question structured explanation, and honest uncalibrated confidence.
"""

from datetime import datetime, timezone, timedelta
import pytest
import pandas as pd

from skyguard_ml.evidence_fusion import fuse_evidence, check_deterministic_overrides
from skyguard_ml.spatial_analysis import analyze_spatial_context
from skyguard_ml.forecast_analysis import analyze_forecast_context
from skyguard_ml.event_context import (
    classify_event_context,
    check_psychrometric_coherence,
    check_isolated_sensor_fault,
)
from skyguard_ml.anomaly_types import infer_anomaly_type, infer_affected_parameter
from skyguard_ml.explainability import build_structured_explanation, build_reason_codes
from skyguard_ml.inference import SkyGuard
from skyguard_ml.canonical_schema import CanonicalTelemetryInput
from skyguard_ml.canonical_pipeline import CanonicalPipelineService


def test_unavailable_spatial_context_neutral():
    """Missing spatial context must yield None score and re-allocate weight without penalty."""
    obs = {
        "station_id": "STN_TEST_01",
        "timestamp": "2026-09-15T12:00:00Z",
        "latitude": 17.3850,
        "longitude": 78.4867,
        "temperature": 32.5,
        "humidity": 60.0,
        "pressure": 1012.0,
    }
    # No nearby context
    spatial = analyze_spatial_context(
        observation=obs,
        nearby_context=None,
        variables=["temperature", "humidity", "pressure"],
    )
    assert spatial["status"] == "SPATIAL_EVIDENCE_UNAVAILABLE"
    assert spatial["spatial_score"] is None
    assert spatial["number_of_nearby_stations"] == 0

    # Fusion without spatial
    evidence = {
        "data_quality": 0.0,
        "statistical": 0.1,
        "temporal": 0.1,
        "ml": 0.1,
        "multivariate": 0.1,
        "spatial": spatial["spatial_score"],  # None
        "forecast": None,
    }
    weights = {
        "data_quality": 0.15,
        "statistical": 0.15,
        "temporal": 0.20,
        "ml": 0.30,
        "multivariate": 0.10,
        "spatial": 0.05,
        "forecast": 0.05,
    }
    fused = fuse_evidence(evidence, weights)
    assert "spatial" in fused["unavailable_sources"]
    assert "spatial" not in fused["used_sources"]
    assert fused["channels"]["spatial"]["status"] == "UNAVAILABLE"
    assert fused["channels"]["spatial"]["weight"] == 0.0
    # Score should reflect only available sources normalized to 1.0
    # (0.0*0.15 + 0.1*(0.15+0.20+0.30+0.10)) / 0.90 = 0.075 / 0.90 = 0.0833
    assert fused["score"] == pytest.approx(0.0833, abs=0.01)


def test_stale_spatial_context_filtered():
    """Nearby station data > 3 hours old must be filtered out as stale."""
    obs_time = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    obs = {
        "station_id": "STN_TARGET",
        "timestamp": obs_time.isoformat(),
        "latitude": 17.3850,
        "longitude": 78.4867,
        "temperature": 30.0,
    }

    stale_peer = {
        "station_id": "STN_PEER_STALE",
        "timestamp": (obs_time - timedelta(hours=4)).isoformat(),  # 4 hours old
        "latitude": 17.4000,
        "longitude": 78.5000,
        "temperature": 30.5,
    }

    spatial = analyze_spatial_context(
        observation=obs,
        nearby_context=[stale_peer],
        variables=["temperature"],
        max_freshness_seconds=10800.0,
    )
    assert spatial["status"] == "SPATIAL_EVIDENCE_UNAVAILABLE"
    assert spatial["spatial_score"] is None
    assert spatial["stale_station_count"] == 1
    assert spatial["fresh_station_count"] == 0


def test_stale_forecast_context_filtered():
    """External forecast data > 3 hours old must be marked UNAVAILABLE."""
    obs_time = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    obs = {
        "timestamp": obs_time.isoformat(),
        "temperature": 32.0,
    }
    stale_forecast = {
        "timestamp": (obs_time - timedelta(hours=3, minutes=30)).isoformat(),
        "temperature": {"value": 31.5},
    }
    fc = analyze_forecast_context(
        observation=obs,
        forecast_context=stale_forecast,
        variables=["temperature"],
        max_freshness_seconds=10800.0,
    )
    assert fc["status"] == "UNAVAILABLE"
    assert fc["forecast_score"] is None
    assert "stale" in fc["details"]["reason"].lower()


def test_conflicting_evidence_detection():
    """Divergence between sensor channels and external context must trigger conflict and INSUFFICIENT_EVIDENCE."""
    # Sensor channels flag high anomaly, while spatial/forecast flag strong normalcy
    evidence = {
        "data_quality": 0.0,
        "statistical": 0.80,
        "temporal": 0.85,
        "ml": 0.90,
        "multivariate": 0.75,
        "spatial": 0.05,     # High agreement/normal
        "forecast": 0.10,    # High match
    }
    weights = {
        "data_quality": 0.10,
        "statistical": 0.15,
        "temporal": 0.15,
        "ml": 0.30,
        "multivariate": 0.15,
        "spatial": 0.075,
        "forecast": 0.075,
    }
    fused = fuse_evidence(evidence, weights)
    assert fused["has_conflict"] is True
    assert fused["status"] == "CONFLICTING_EVIDENCE"
    assert "Evidence divergence" in fused["conflict_details"]

    classification = classify_event_context(
        anomaly_score=fused["score"],
        historical_score=0.80,
        temporal_score=0.85,
        multivariate_score=0.75,
        spatial_score=0.05,
        forecast_score=0.10,
        has_conflict=True,
    )
    assert classification == "INSUFFICIENT_EVIDENCE"


def test_deterministic_quality_override():
    """Physical limit violations must override ML score to 1.0 immediately."""
    corrupted_obs = {
        "temperature": 75.0,  # Exceeds +65°C atmospheric maximum
        "humidity": 45.0,
        "pressure": 1010.0,
    }
    is_override, o_type, reason = check_deterministic_overrides(corrupted_obs, quality_flags=[])
    assert is_override is True
    assert o_type == "DATA_QUALITY_OVERRIDE"
    assert "physical atmospheric bounds" in reason

    # Empty payload override
    empty_obs = {
        "temperature": None,
        "humidity": None,
        "pressure": None,
    }
    is_override, o_type, reason = check_deterministic_overrides(empty_obs, quality_flags=[])
    assert is_override is True
    assert o_type == "COMMUNICATION_OVERRIDE"

    # Fused result with override
    fused = fuse_evidence(
        evidence={"data_quality": 1.0, "ml": 0.05},
        weights={"data_quality": 0.5, "ml": 0.5},
        observation=corrupted_obs,
    )
    assert fused["is_override"] is True
    assert fused["score"] == 1.0
    assert fused["status"] == "DATA_QUALITY_OVERRIDE"


def test_single_vs_persistent_anomaly():
    """Single observation anomaly is transient; consecutive >= 3 is persistent."""
    evidence = {"ml": 0.8, "temporal": 0.8}
    weights = {"ml": 0.5, "temporal": 0.5}

    fused_single = fuse_evidence(evidence, weights, anomaly_run_length=1)
    assert fused_single["persistence"]["is_persistent"] is False
    assert fused_single["persistence"]["classification"] == "TRANSIENT_ANOMALY"

    fused_persistent = fuse_evidence(evidence, weights, anomaly_run_length=3)
    assert fused_persistent["persistence"]["is_persistent"] is True
    assert fused_persistent["persistence"]["classification"] == "PERSISTENT_ANOMALY"


def test_psychrometric_weather_vs_sensor_reasoning():
    """Coherent thermodynamic delta classifies as LIKELY_WEATHER_EVENT; isolated delta as LIKELY_SENSOR_ANOMALY."""
    # Thunderstorm gust front: Temp drop >= 2.5°C with Humidity surge >= 5%
    coherent_deltas = {"temperature": -3.5, "humidity": 12.0, "pressure": -1.5}
    is_weather, w_desc = check_psychrometric_coherence(coherent_deltas)
    assert is_weather is True
    assert "Thermodynamic coherence" in w_desc

    class_weather = classify_event_context(
        anomaly_score=0.70,
        historical_score=0.50,
        temporal_score=0.75,
        multivariate_score=0.65,
        spatial_score=0.20,
        spatial_status="SPATIAL_CORROBORATED",
        deltas=coherent_deltas,
    )
    assert class_weather == "LIKELY_WEATHER_EVENT"

    # Isolated sensor excursion: Temp jumps +8°C while RH and Pressure remain invariant
    isolated_deltas = {"temperature": 8.0, "humidity": 0.2, "pressure": 0.1}
    is_isolated, s_desc = check_isolated_sensor_fault(isolated_deltas)
    assert is_isolated is True
    assert "Isolated temperature excursion" in s_desc

    class_sensor = classify_event_context(
        anomaly_score=0.75,
        historical_score=0.60,
        temporal_score=0.80,
        multivariate_score=0.70,
        spatial_score=0.70,
        spatial_status="SPATIAL_NOT_CORROBORATED",
        deltas=isolated_deltas,
    )
    assert class_sensor == "LIKELY_SENSOR_ANOMALY"


def test_authoritative_waterfall_and_truthful_confidence():
    """Verify machine-readable waterfall, 5-question structured explanation, and uncalibrated confidence."""
    from pathlib import Path
    art_dir = Path(__file__).resolve().parent.parent / "skyguard_ml" / "artifacts"
    engine = SkyGuard(artifact_dir=art_dir).load()
    obs = {
        "station_id": "STN_WF_01",
        "timestamp": "2026-09-15T12:00:00Z",
        "temperature": 45.0,  # Elevated
        "humidity": 20.0,
        "pressure": 1008.0,
    }
    history = [
        {"station_id": "STN_WF_01", "timestamp": f"2026-09-15T{h:02d}:00:00Z", "temperature": 30.0, "humidity": 60.0, "pressure": 1013.0}
        for h in range(1, 12)
    ]

    res = engine.predict(
        observation=obs,
        historical_context=history,
    )

    # 1. Uncalibrated confidence status
    assert res["confidence_status"] == "NOT_CALIBRATED"

    # 2. Affected parameter identified
    assert res["affected_parameter"] in ("temperature", "multiple", "unknown")

    # 3. Structured 5-question explanation
    structured = res["structured_explanation"]
    assert "what_happened" in structured
    assert "why_flagged" in structured
    assert "supporting_evidence" in structured
    assert "contradicting_evidence" in structured
    assert "missing_evidence" in structured

    # 4. Sequential machine-readable waterfall
    waterfall = res["evidence_waterfall"]
    assert len(waterfall) >= 8
    stages = [step["stage"] for step in waterfall]
    assert "OBSERVATION" in stages
    assert "DATA_QUALITY" in stages
    assert "TEMPORAL" in stages
    assert "STATISTICAL" in stages
    assert "ML_DETECTOR" in stages
    assert "MULTIVARIATE" in stages
    assert "SPATIAL_CONTEXT" in stages
    assert "EVIDENCE_FUSION" in stages
    assert "CLASSIFICATION" in stages

    for step in waterfall:
        assert "stage" in step
        assert "status" in step
        assert "summary" in step
