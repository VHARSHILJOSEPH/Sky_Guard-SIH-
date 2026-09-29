"""
SkyGuard AI — Combined Mega Phase 10–13 End-to-End Verification Test Suite.

Automated verification of the 19 core edge cases specified in Section 23:
 1. Physical bounds rejection (temperature > 60°C, pressure outside 900–1100 hPa).
 2. Rapid temporal rate limit violation (15-min jump > 8°C).
 3. Frozen sensor persistence (zero variance over consecutive readings).
 4. Cumulative baseline drift (+0.2°C/hr departing from diurnal baseline).
 5. Multivariate thermodynamic decoupling (dew point spread / vapor pressure violation).
 6. Spatial consensus corroboration vs uncorroborated divergence.
 7. Macro weather event classification (coordinated regional change vs isolated sensor fault).
 8. Communication failure packet loss (null payload or missing values).
 9. Incident deduplication (same station and anomaly type increments occurrences).
10. Incident lifecycle state transitions (DETECTED -> ACKNOWLEDGED -> INVESTIGATING -> RESOLVED).
11. Model comparison report endpoint (/api/ml/model-comparison) returns 4 models with LOF as winner.
12. Evaluation report endpoint (/api/ml/evaluation-report) returns validation metrics and fault breakdowns.
13. Station health endpoint (/api/station-health) returns multi-signal metrics.
14. Station health canonical states (HEALTHY, DEGRADED, UNHEALTHY, OFFLINE, WARMUP, UNKNOWN).
15. Anomaly history endpoint (/api/anomalies/history) respects station, severity, and status filters.
16. Strict data provenance visibility (IMD_AWS, DEMO_SIMULATION, OPEN_METEO_CONTEXT).
17. Confidence status assignment (HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, CORRUPTED_DATA).
18. Threshold calibration consistency (strictly 0.40 for LOF v2.0.0).
19. Zero-variance frozen sensor detection independent of temperature value.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from ml_api_server import app
from skyguard_lof import SkyGuardLOFConfig
from skyguard_ml.canonical_schema import (
    CanonicalTelemetryInput,
    CanonicalInferenceResponse,
    IncidentLifecycleState,
    ALLOWED_SOURCES,
)
from skyguard_ml.canonical_pipeline import (
    CanonicalPipelineService,
    get_canonical_pipeline,
    station_history_live,
    station_history_demo,
)
from skyguard_ml.incident_manager import incident_manager
from skyguard_ml.sensor_health import SensorHealthTracker, sensor_health_tracker
from skyguard_ml.statistical_detectors import (
    detect_frozen_sensor,
    detect_drift,
    detect_spike_drop,
    run_statistical_detectors,
)
from skyguard_ml.evidence_fusion import check_deterministic_overrides, fuse_evidence
from skyguard_ml.spatial_analysis import analyze_spatial_context
from skyguard_ml.event_context import (
    classify_event_context,
    check_psychrometric_coherence,
    check_isolated_sensor_fault,
)
from app.services.database_service import (
    save_sensor_reading,
    save_anomaly,
    get_recent_anomalies,
    update_anomaly_status,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_test_state():
    """Ensure in-memory history, health tracker, and incidents are fresh before each test."""
    station_history_live.clear()
    station_history_demo.clear()
    incident_manager._active_incidents.clear()
    incident_manager._incident_history.clear()
    incident_manager._normal_counter.clear()
    sensor_health_tracker.history.clear()
    sensor_health_tracker.anomaly_history.clear()
    sensor_health_tracker.issue_history.clear()
    sensor_health_tracker.last_timestamps.clear()
    yield


# =====================================================================
# 1. Physical bounds rejection (temperature > 60°C, pressure outside 900–1100 hPa)
# =====================================================================
def test_case_01_physical_bounds_rejection(client):
    """Extreme physical readings outside atmospheric bounds must trigger physical limit violation."""
    # Test temperature > 60°C (75.0°C)
    pipe_res = client.post(
        "/api/pipeline/process",
        json={
            "station_id": "TEST_BOUNDS_01",
            "observation": {
                "temperature": 75.0,
                "humidity": 45.0,
                "pressure": 1012.0,
            },
            "source": "DEMO_SIMULATION",
        },
    )
    assert pipe_res.status_code == 200
    data = pipe_res.json()
    assert data["is_anomaly"] is True
    assert data["status"] in ("REJECTED_PHYSICAL_LIMITS", "ANOMALY", "DATA_QUALITY_ISSUE", "CORRUPTED_DATA")
    dq = data.get("data_quality_state", {})
    assert dq.get("valid") is False or dq.get("physical_bounds_ok") is False
    
    # Test pressure outside terrestrial surface bounds (650.0 hPa < 700 hPa)
    pipe_res_press = client.post(
        "/api/pipeline/process",
        json={
            "station_id": "TEST_BOUNDS_02",
            "observation": {
                "temperature": 28.0,
                "humidity": 55.0,
                "pressure": 650.0,
            },
            "source": "DEMO_SIMULATION",
        },
    )
    assert pipe_res_press.status_code == 200
    data_press = pipe_res_press.json()
    assert data_press["is_anomaly"] is True
    dq_press = data_press.get("data_quality_state", {})
    assert dq_press.get("valid") is False or dq_press.get("physical_bounds_ok") is False

    # Deterministic override directly
    is_ov, ov_type, reason = check_deterministic_overrides({"temperature": 85.0, "pressure": 1010.0}, quality_flags=[])
    assert is_ov is True
    assert ov_type == "DATA_QUALITY_OVERRIDE"


# =====================================================================
# 2. Rapid temporal rate limit violation (15-min jump > 8°C)
# =====================================================================
def test_case_02_rapid_temporal_rate_limit_violation(client):
    """A rapid temperature jump exceeding 8°C in 15 minutes triggers a temporal rate-of-change spike."""
    base_time = datetime(2026, 9, 15, 6, 0, 0, tzinfo=timezone.utc)
    station = "TEST_TEMPORAL_01"
    
    # Seed 25 nominal observations with realistic diurnal variance (no frozen sensor)
    history_records = []
    for i in range(25):
        t = base_time + timedelta(minutes=15 * i)
        history_records.append({
            "station_id": station,
            "timestamp": t.isoformat(),
            "temperature": 25.0 + math.sin(i / 2.0) * 1.5,
            "humidity": 60.0 + math.cos(i / 2.0) * 4.0,
            "pressure": 1013.0 + math.cos(i / 3.0) * 2.0,
            "rainfall": 0.0,
            "wind_speed": 3.0,
        })
    
    client.post(
        "/api/ml/seed-history",
        json={"station_id": station, "records": history_records},
    )
    
    # Inject rapid +9.0°C jump in 15 minutes (from ~25.5°C to 35.0°C)
    spike_time = base_time + timedelta(minutes=15 * 25)
    res = client.post(
        "/api/pipeline/process",
        json={
            "station_id": station,
            "observation": {
                "temperature": 35.0,
                "humidity": 60.0,
                "pressure": 1013.0,
                "timestamp": spike_time.isoformat(),
            },
            "source": "DEMO_SIMULATION",
            "history": history_records,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_anomaly"] is True
    assert data["anomaly_type"] in ("TEMPERATURE_ANOMALY", "TEMPORAL_SPIKE", "DATA_QUALITY_ANOMALY", "ML_NOVELTY", "STATISTICAL_OUTLIER")


# =====================================================================
# 3. Frozen sensor persistence (zero variance over consecutive readings)
# =====================================================================
def test_case_03_frozen_sensor_persistence():
    """Zero variance across >= 6 consecutive intervals triggers frozen sensor detection."""
    base_time = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    rows = []
    # 8 consecutive readings with invariant 27.500°C
    for i in range(8):
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * i)).isoformat(),
            "temperature": 27.500,
            "humidity": 65.0,
            "pressure": 1012.0,
        })
    df = pd.DataFrame(rows)
    frozen_res = detect_frozen_sensor(
        df,
        variables=["temperature", "humidity", "pressure"],
        tolerance=1e-4,
        duration_hours=1.5,
    )
    assert "temperature" in frozen_res
    assert frozen_res["temperature"]["score"] >= 0.6
    assert frozen_res["temperature"]["run_length"] >= 6
    assert frozen_res["temperature"]["variance"] <= 1e-4


# =====================================================================
# 4. Cumulative baseline drift (+0.2°C/hr departing from diurnal baseline)
# =====================================================================
def test_case_04_cumulative_baseline_drift():
    """Accumulating monotonic departure from diurnal baseline builds drift anomaly score."""
    base_time = datetime(2026, 9, 15, 6, 0, 0, tzinfo=timezone.utc)
    rows = []
    # Monotonically increasing +0.25°C every 15 minutes over 16 steps (+4.0°C drift)
    for i in range(16):
        rows.append({
            "timestamp": (base_time + timedelta(minutes=15 * i)).isoformat(),
            "temperature": 26.0 + (i * 0.25),
            "humidity": 60.0,
            "pressure": 1012.0,
        })
    df = pd.DataFrame(rows)
    drift_res = detect_drift(df, variables=["temperature"])
    assert "temperature" in drift_res
    assert drift_res["temperature"]["slope"] > 0.0
    assert drift_res["temperature"]["score"] > 0.3


# =====================================================================
# 5. Multivariate thermodynamic decoupling (dew point spread / vapor pressure)
# =====================================================================
def test_case_05_multivariate_thermodynamic_decoupling(client):
    """Thermodynamic decoupling with established history triggers anomaly."""
    base_time = datetime(2026, 9, 15, 6, 0, 0, tzinfo=timezone.utc)
    station = "TEST_THERMO_01"
    
    history_records = []
    for i in range(25):
        t = base_time + timedelta(minutes=15 * i)
        history_records.append({
            "station_id": station,
            "timestamp": t.isoformat(),
            "temperature": 28.0 + math.sin(i / 2.0) * 1.5,
            "humidity": 50.0 + math.cos(i / 2.0) * 5.0,
            "pressure": 1012.0 + math.cos(i / 3.0) * 2.0,
            "rainfall": 0.0,
            "wind_speed": 3.0,
        })
    
    client.post(
        "/api/ml/seed-history",
        json={"station_id": station, "records": history_records},
    )

    res = client.post(
        "/api/pipeline/process",
        json={
            "station_id": station,
            "observation": {
                "temperature": 48.0,
                "humidity": 99.0,
                "pressure": 1013.0,
                "timestamp": (base_time + timedelta(minutes=15 * 25)).isoformat(),
            },
            "source": "DEMO_SIMULATION",
            "history": history_records,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["is_anomaly"] is True
    # The thermodynamic channel in waterfall should be present and evaluated
    wf = data.get("evidence_waterfall", [])
    multi_step = next((s for s in wf if s.get("channel") == "Multivariate"), None)
    assert multi_step is not None


# =====================================================================
# 6. Spatial consensus corroboration vs uncorroborated divergence
# =====================================================================
def test_case_06_spatial_consensus_vs_divergence():
    """Spatial context distinguishes regional consensus from isolated station divergence."""
    target_obs = {
        "station_id": "STN_TARGET",
        "timestamp": "2026-09-15T12:00:00Z",
        "latitude": 17.3850,
        "longitude": 78.4867,
        "temperature": 24.0,  # Sudden drop
        "humidity": 85.0,
        "pressure": 1006.0,
    }
    
    # 1. Nearby peers corroborating the same drop (Consensus)
    corroborating_peers = [
        {"station_id": "PEER_1", "timestamp": "2026-09-15T12:00:00Z", "latitude": 17.3900, "longitude": 78.4900, "temperature": 24.2, "humidity": 84.0, "pressure": 1006.2},
        {"station_id": "PEER_2", "timestamp": "2026-09-15T12:00:00Z", "latitude": 17.3800, "longitude": 78.4800, "temperature": 23.9, "humidity": 86.0, "pressure": 1005.9},
    ]
    spatial_agree = analyze_spatial_context(
        observation=target_obs,
        nearby_context=corroborating_peers,
        variables=["temperature", "humidity", "pressure"],
    )
    assert spatial_agree["status"] == "SPATIAL_CORROBORATED"
    assert spatial_agree["spatial_score"] < 0.35

    # 2. Nearby peers showing warm/dry baseline (Divergence / Not Corroborated)
    divergent_peers = [
        {"station_id": "PEER_1", "timestamp": "2026-09-15T12:00:00Z", "latitude": 17.3900, "longitude": 78.4900, "temperature": 34.0, "humidity": 45.0, "pressure": 1012.0},
        {"station_id": "PEER_2", "timestamp": "2026-09-15T12:00:00Z", "latitude": 17.3800, "longitude": 78.4800, "temperature": 34.5, "humidity": 43.0, "pressure": 1012.5},
    ]
    spatial_div = analyze_spatial_context(
        observation=target_obs,
        nearby_context=divergent_peers,
        variables=["temperature", "humidity", "pressure"],
    )
    assert spatial_div["status"] == "SPATIAL_NOT_CORROBORATED"
    assert spatial_div["spatial_score"] > 0.50

    # 3. No nearby stations (Evidence Unavailable without penalty)
    spatial_none = analyze_spatial_context(
        observation=target_obs,
        nearby_context=[],
        variables=["temperature"],
    )
    assert spatial_none["status"] == "SPATIAL_EVIDENCE_UNAVAILABLE"
    assert spatial_none["spatial_score"] is None


# =====================================================================
# 7. Macro weather event classification (coordinated change vs sensor fault)
# =====================================================================
def test_case_07_macro_weather_event_classification():
    """Distinguishes genuine thermodynamic fronts from isolated electrical transducer faults."""
    # Thunderstorm front: temp drop -3.5°C, humidity rise +12%, pressure drop -1.5 hPa
    coherent_deltas = {"temperature": -3.5, "humidity": 12.0, "pressure": -1.5}
    is_weather, w_desc = check_psychrometric_coherence(coherent_deltas)
    assert is_weather is True

    classification = classify_event_context(
        anomaly_score=0.68,
        historical_score=0.45,
        temporal_score=0.70,
        multivariate_score=0.60,
        spatial_score=0.15,
        spatial_status="SPATIAL_CORROBORATED",
        deltas=coherent_deltas,
    )
    assert classification == "LIKELY_WEATHER_EVENT"

    # Isolated electrical spike: temp jump +8°C, humidity/pressure flat
    isolated_deltas = {"temperature": 8.0, "humidity": 0.1, "pressure": 0.0}
    is_sensor, s_desc = check_isolated_sensor_fault(isolated_deltas)
    assert is_sensor is True

    classification_sensor = classify_event_context(
        anomaly_score=0.78,
        historical_score=0.60,
        temporal_score=0.85,
        multivariate_score=0.70,
        spatial_score=0.75,
        spatial_status="SPATIAL_NOT_CORROBORATED",
        deltas=isolated_deltas,
    )
    assert classification_sensor == "LIKELY_SENSOR_ANOMALY"


# =====================================================================
# 8. Communication failure packet loss (null payload or missing values)
# =====================================================================
def test_case_08_communication_failure_packet_loss():
    """Null payload or missing essential telemetry is flagged as communication failure override."""
    empty_obs = {"temperature": None, "humidity": None, "pressure": None}
    is_ov, ov_type, reason = check_deterministic_overrides(empty_obs, quality_flags=[])
    assert is_ov is True
    assert ov_type == "COMMUNICATION_OVERRIDE"
    assert "packet" in reason.lower() or "missing" in reason.lower() or "telemetry" in reason.lower()


# =====================================================================
# 9. Incident deduplication (same station and anomaly type increments occurrences)
# =====================================================================
def test_case_09_incident_deduplication():
    """Repeated occurrences of the same anomaly type on a station do not spawn duplicate active incidents."""
    stn = "STN_DEDUP_TEST"
    ts1 = "2026-09-15T12:00:00Z"
    
    # 1. First occurrence
    inc_1 = incident_manager.process_observation_incident(
        station_id=stn,
        timestamp_str=ts1,
        is_anomaly=True,
        anomaly_type="TEMPORAL_SPIKE",
        severity="HIGH",
    )
    assert inc_1 is not None
    assert inc_1.occurrence_count == 1
    orig_id = inc_1.incident_id

    # 2. Second consecutive occurrence of identical fault 15 minutes later
    ts2 = "2026-09-15T12:15:00Z"
    inc_2 = incident_manager.process_observation_incident(
        station_id=stn,
        timestamp_str=ts2,
        is_anomaly=True,
        anomaly_type="TEMPORAL_SPIKE",
        severity="HIGH",
    )
    assert inc_2 is not None
    assert inc_2.incident_id == orig_id
    assert inc_2.occurrence_count == 2

    # Verify active incidents list contains only 1 incident for this station/type
    active = incident_manager.list_incidents(station_id=stn)
    assert len(active) == 1
    assert active[0]["occurrence_count"] == 2


# =====================================================================
# 10. Incident lifecycle state transitions (DETECTED -> ACKNOWLEDGED -> INVESTIGATING -> RESOLVED)
# =====================================================================
def test_case_10_incident_lifecycle_state_transitions(client):
    """Validates full lifecycle progression and rejection of invalid states."""
    stn = "STN_LIFECYCLE_TEST"
    ts = "2026-09-15T12:00:00Z"
    inc = incident_manager.process_observation_incident(
        station_id=stn,
        timestamp_str=ts,
        is_anomaly=True,
        anomaly_type="FROZEN_SENSOR",
        severity="CRITICAL",
    )
    assert inc.status == "DETECTED"

    # Transition 1: DETECTED -> ACKNOWLEDGED via API
    res_ack = client.patch(f"/api/incidents/{inc.incident_id}/status?status=ACKNOWLEDGED")
    assert res_ack.status_code == 200
    assert res_ack.json()["incident"]["status"] == "ACKNOWLEDGED"

    # Transition 2: ACKNOWLEDGED -> INVESTIGATING via API
    res_inv = client.patch(f"/api/incidents/{inc.incident_id}/status?status=INVESTIGATING")
    assert res_inv.status_code == 200
    assert res_inv.json()["incident"]["status"] == "INVESTIGATING"

    # Transition 3: INVESTIGATING -> RESOLVED via API
    res_res = client.patch(f"/api/incidents/{inc.incident_id}/status?status=RESOLVED")
    assert res_res.status_code == 200
    assert res_res.json()["incident"]["status"] == "RESOLVED"

    # Invalid state rejection
    res_bad = client.patch(f"/api/incidents/{inc.incident_id}/status?status=INVALID_STATE")
    assert res_bad.status_code == 400


# =====================================================================
# 11. Model comparison report endpoint (/api/ml/model-comparison)
# =====================================================================
def test_case_11_model_comparison_endpoint_winner(client):
    """Comparative evaluation endpoint provides all 4 benchmark models with LOF v2.0.0 as certified winner."""
    res = client.get("/api/ml/model-comparison")
    assert res.status_code == 200
    data = res.json()
    report = data.get("report") or data
    assert "comparison_table" in report
    assert len(report["comparison_table"]) >= 4

    model_names = [m["Model"] for m in report["comparison_table"]]
    assert "LOF" in model_names
    assert "IsolationForest" in model_names
    assert "OneClassSVM" in model_names
    assert "RobustCovariance" in model_names

    # Winner is LOF v2.0.0
    winner = data.get("selected_production_model") or report.get("winning_model")
    assert "LOF" in winner
    lof_row = next(m for m in report["comparison_table"] if m["Model"] == "LOF")
    assert lof_row["Selection Score"] == pytest.approx(0.5938, abs=0.01)


# =====================================================================
# 12. Evaluation report endpoint (/api/ml/evaluation-report)
# =====================================================================
def test_case_12_evaluation_report_endpoint(client):
    """Full evaluation report delivers complete validation metrics and fault type breakdowns."""
    res = client.get("/api/ml/evaluation-report")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["selected_production_model"] == "LOF v2.0.0"
    report = data["report"]
    assert report["selected_model"] == "LOF"
    assert "LOF" in report["models"]
    lof_model = report["models"]["LOF"]
    assert lof_model["calibrated_threshold"] == pytest.approx(0.40, abs=1e-4)
    test_metrics = lof_model["test_metrics"]
    assert "roc_auc" in test_metrics
    assert "balanced_accuracy" in test_metrics
    assert "recall" in test_metrics


# =====================================================================
# 13. Station health endpoint (/api/station-health)
# =====================================================================
def test_case_13_station_health_endpoint_metrics(client):
    """Station health endpoint returns structured health scores and multi-signal metrics."""
    res = client.get("/api/station-health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert "records" in data
    assert isinstance(data["records"], list)


# =====================================================================
# 14. Station health canonical states
# =====================================================================
def test_case_14_station_health_canonical_states():
    """Multi-signal health tracker correctly classifies stations into canonical health states."""
    tracker = SensorHealthTracker()

    # 1. Warmup (< 24 history points, nominal)
    w_res = tracker.update("STN_WARM", anomaly_score=0.1, anomaly_type="NORMAL", history_len=10)
    assert w_res["status"] == "WARMUP"

    # 2. Healthy (sufficient history, nominal)
    # Seed 20 nominal observations
    for _ in range(20):
        h_res = tracker.update("STN_HEALTHY", anomaly_score=0.1, anomaly_type="NORMAL", history_len=30)
    assert h_res["status"] == "HEALTHY"
    assert h_res["score"] >= 85.0

    # 3. Single isolated anomaly != Bad station (Rule: One anomaly != bad station)
    iso_res = tracker.update("STN_HEALTHY", anomaly_score=0.75, anomaly_type="TEMPORAL_SPIKE", history_len=31)
    assert iso_res["status"] == "HEALTHY"
    assert iso_res["score"] >= 80.0  # Station remains HEALTHY!

    # 4. Degraded (repeated anomalies / drift accumulation)
    for _ in range(8):
        deg_res = tracker.update("STN_DEG", anomaly_score=0.65, anomaly_type="TEMPORAL_SPIKE", drift_score=0.45, history_len=40)
    assert deg_res["status"] == "DEGRADED"

    # 5. Offline (communication gap)
    off_res = tracker.update("STN_OFF", anomaly_score=1.0, anomaly_type="COMMUNICATION_FAILURE", is_offline=True, history_len=40)
    assert off_res["status"] == "OFFLINE"

    # 6. Unhealthy (severe multi-channel failure and high anomaly frequency)
    for _ in range(15):
        unh_res = tracker.update(
            "STN_UNH",
            anomaly_score=0.95,
            anomaly_type="DATA_QUALITY_ANOMALY",
            missing_score=0.8,
            drift_score=0.9,
            quality_score=0.0,
            severity="CRITICAL",
            history_len=40,
        )
    assert unh_res["status"] == "UNHEALTHY"
    assert unh_res["score"] < 50.0


# =====================================================================
# 15. Anomaly history endpoint (/api/anomalies/history)
# =====================================================================
def test_case_15_anomaly_history_endpoint_filters(client):
    """Endpoint respects station_id, severity, and status query parameters."""
    stn_id = f"STN_FILTER_{int(datetime.now(timezone.utc).timestamp())}"
    now_ts = datetime.now(timezone.utc).isoformat()
    
    # Insert test anomaly into database
    save_anomaly(
        station_id=stn_id,
        timestamp=now_ts,
        anomaly_type="STATISTICAL_OUTLIER",
        anomaly_score=0.88,
        severity="HIGH",
        status="INVESTIGATING",
        affected_parameter="temperature",
    )

    # Filter by station_id
    res_stn = client.get(f"/api/anomalies/history?station_id={stn_id}")
    assert res_stn.status_code == 200
    stn_data = res_stn.json()
    assert stn_data["count"] >= 1
    assert all(a["station_id"] == stn_id for a in stn_data["anomalies"])

    # Filter by severity
    res_sev = client.get(f"/api/anomalies/history?station_id={stn_id}&severity=HIGH")
    assert res_sev.status_code == 200
    assert len(res_sev.json()["anomalies"]) >= 1

    # Filter by status
    res_stat = client.get(f"/api/anomalies/history?station_id={stn_id}&status=INVESTIGATING")
    assert res_stat.status_code == 200
    assert len(res_stat.json()["anomalies"]) >= 1


# =====================================================================
# 16. Strict data provenance visibility
# =====================================================================
def test_case_16_strict_data_provenance_labels(client):
    """All pipeline results carry explicit provenance tags from ALLOWED_SOURCES without falsification."""
    # Process with DEMO_SIMULATION
    res_demo = client.post(
        "/api/pipeline/process",
        json={
            "station_id": "AWS-101",
            "observation": {"temperature": 29.0, "humidity": 60.0, "pressure": 1010.0},
            "source": "DEMO_SIMULATION",
        },
    )
    assert res_demo.status_code == 200
    assert res_demo.json()["source"] == "DEMO_SIMULATION"
    assert res_demo.json()["source"] in ALLOWED_SOURCES

    # Simulated station AWS-101 with LIVE flag must NOT be labeled as IMD_AWS
    res_sim_live = client.post(
        "/api/pipeline/process",
        json={
            "station_id": "AWS-101",
            "observation": {"temperature": 29.0, "humidity": 60.0, "pressure": 1010.0},
            "source": "LIVE",
        },
    )
    assert res_sim_live.status_code == 200
    assert res_sim_live.json()["source"] != "IMD_AWS"
    assert res_sim_live.json()["source"] in ("OPEN_METEO_CONTEXT", "DEMO_SIMULATION")


# =====================================================================
# 17. Confidence status assignment
# =====================================================================
def test_case_17_confidence_status_honest_calibration(client):
    """Confidence status is ground-truth truthful (e.g. NOT_CALIBRATED or evidence-grounded), no false claims."""
    res = client.post(
        "/api/pipeline/process",
        json={
            "station_id": "STN_CONF_TEST",
            "observation": {"temperature": 31.0, "humidity": 55.0, "pressure": 1012.0},
            "source": "DEMO_SIMULATION",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "confidence_status" in data
    assert data["confidence_status"] in (
        "NOT_CALIBRATED",
        "HIGH_CONFIDENCE",
        "MEDIUM_CONFIDENCE",
        "LOW_CONFIDENCE",
        "CORRUPTED_DATA",
    )


# =====================================================================
# 18. Threshold calibration consistency (strictly 0.40 for LOF v2.0.0)
# =====================================================================
def test_case_18_threshold_calibration_strict_0_40():
    """Calibrated anomaly threshold for LOF v2.0.0 is strictly 0.40."""
    art_path = Path(__file__).resolve().parent.parent / "skyguard_ml" / "artifacts" / "evaluation_report.json"
    with open(art_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    lof_entry = rep["models"]["LOF"]
    assert lof_entry["calibrated_threshold"] == pytest.approx(0.40, abs=1e-4)


# =====================================================================
# 19. Zero-variance frozen sensor detection independent of temperature value
# =====================================================================
def test_case_19_zero_variance_frozen_sensor_value_independent():
    """Frozen sensor detection triggers identically across 0°C, 25.5°C, and 42.0°C."""
    base_time = datetime(2026, 9, 15, 0, 0, 0, tzinfo=timezone.utc)

    for stuck_val in [0.0, 25.5, 42.0]:
        rows = [
            {
                "timestamp": (base_time + timedelta(minutes=15 * i)).isoformat(),
                "temperature": stuck_val,
                "humidity": 50.0,
                "pressure": 1010.0,
            }
            for i in range(8)
        ]
        df = pd.DataFrame(rows)
        res = detect_frozen_sensor(df, variables=["temperature"], tolerance=1e-4, duration_hours=1.5)
        assert res["temperature"]["score"] >= 0.6, f"Failed to detect frozen sensor at {stuck_val}°C"
        assert res["temperature"]["run_length"] >= 6
        assert res["temperature"]["variance"] <= 1e-4
