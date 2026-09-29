"""
SkyGuard AI — Canonical Pipeline Automated Verification Test Suite.

Validates Requirements (Tasks 2-14):
1. Canonical telemetry schema validation & alias normalization.
2. Timestamp validation & UTC ISO-8601 formatting.
3. Strict source labeling taxonomy (IMD_AWS, DEMO_SIMULATION, OPEN_METEO_CONTEXT, OFFLINE_PREVIEW).
4. Demo vs Live isolation (separate buffers, preventing contamination).
5. 7-Channel structured inference response (no probability claims).
6. Warmup state (< 24 history points).
7. Insufficient evidence and explicit classification states.
8. Anomaly incident lifecycle transitions (DETECTED, ACKNOWLEDGED, INVESTIGATING, RESOLVED).
9. Alert deduplication across consecutive observations of the same fault.
10. Database persistence hydration across server restarts.
11. Model version and schema propagation.
12. FastAPI POST /api/telemetry, GET /api/incidents, and PATCH /api/incidents/{id}/status endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
import pytest
from fastapi.testclient import TestClient

from skyguard_ml.canonical_schema import (
    CanonicalTelemetryInput,
    CanonicalInferenceResponse,
    ALLOWED_SOURCES,
    parse_and_validate_timestamp,
)
from skyguard_ml.canonical_pipeline import (
    CanonicalPipelineService,
    get_canonical_pipeline,
    station_history_live,
    station_history_demo,
)
from skyguard_ml.incident_manager import IncidentManager, incident_manager
from ml_api_server import app, get_model, get_engine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clean_test_state():
    """Ensure in-memory history and incidents are fresh before each test."""
    station_history_live.clear()
    station_history_demo.clear()
    incident_manager._active_incidents.clear()
    incident_manager._incident_history.clear()
    incident_manager._normal_counter.clear()
    yield


# =====================================================================
# 1. Telemetry Validation & Alias Normalization
# =====================================================================

def test_canonical_telemetry_normalization():
    """Validates alias mapping (temp -> temperature_c, hum -> humidity_pct, baro -> pressure_hpa)."""
    raw_payload = {
        "station_id": "TEST_AWS_01",
        "timestamp": "2026-09-14T10:00:00Z",
        "temp": 28.5,
        "hum": 72.0,
        "baro": 1012.3,
        "WS": 3.4,
        "RAIN": 0.0,
        "source": "LIVE",
    }
    canonical = CanonicalTelemetryInput(**raw_payload)
    assert canonical.temperature_c == 28.5
    assert canonical.humidity_pct == 72.0
    assert canonical.pressure_hpa == 1012.3
    assert canonical.wind_speed_ms == 3.4
    assert canonical.rainfall_mm == 0.0
    # LIVE normalized to IMD_AWS
    assert canonical.source == "IMD_AWS"


def test_canonical_telemetry_sanitizes_nan():
    """Validates NaN, null, and non-finite numbers are converted to None."""
    payload = {
        "station_id": "TEST_AWS_01",
        "timestamp": "2026-09-14T10:00:00Z",
        "temperature_c": "NaN",
        "humidity_pct": None,
        "pressure_hpa": "1008.5",
        "source": "IMD_AWS",
    }
    canonical = CanonicalTelemetryInput(**payload)
    assert canonical.temperature_c is None
    assert canonical.humidity_pct is None
    assert canonical.pressure_hpa == 1008.5


# =====================================================================
# 2. Timestamp Validation
# =====================================================================

def test_timestamp_validation_valid_formats():
    """Validates ISO UTC parsing and normalization."""
    # Standard ISO UTC
    ts1 = parse_and_validate_timestamp("2026-09-14T12:00:00Z")
    assert ts1 == "2026-09-14T12:00:00Z"

    # With offset
    ts2 = parse_and_validate_timestamp("2026-09-14T17:30:00+05:30")
    assert ts2 == "2026-09-14T12:00:00Z"


def test_timestamp_validation_rejects_invalid():
    """Validates invalid timestamps or unphysical years are rejected."""
    with pytest.raises(ValueError):
        parse_and_validate_timestamp("invalid-date-string")

    with pytest.raises(ValueError):
        # Year outside 1990-2100 meteorological range
        parse_and_validate_timestamp("1820-01-01T00:00:00Z")


# =====================================================================
# 3. Source Labeling Taxonomy
# =====================================================================

def test_source_labeling_allowed_sources():
    """Ensures only approved sources are allowed."""
    for s in ("IMD_AWS", "DEMO_SIMULATION", "OPEN_METEO_CONTEXT", "OFFLINE_PREVIEW"):
        inp = CanonicalTelemetryInput(
            station_id="TEST_01",
            timestamp="2026-09-14T10:00:00Z",
            temperature_c=25.0,
            humidity_pct=60.0,
            pressure_hpa=1010.0,
            source=s,
        )
        assert inp.source == s


def test_source_labeling_rejects_fake_sources():
    """Ensures arbitrary invalid source names are rejected."""
    with pytest.raises(ValueError) as exc:
        CanonicalTelemetryInput(
            station_id="TEST_01",
            timestamp="2026-09-14T10:00:00Z",
            source="FAKE_UNVERIFIED_FEED",
        )
    assert "Invalid source" in str(exc.value)


# =====================================================================
# 4. Demo vs Live Isolation
# =====================================================================

def test_demo_live_isolation():
    """Ensures demo simulation observations never pollute live station history or previous observations."""
    pipeline = get_canonical_pipeline()
    station_id = f"ISOL_STN_{uuid.uuid4().hex[:8]}"

    # Ingest Live Observation
    live_input = CanonicalTelemetryInput(
        station_id=station_id,
        timestamp="2026-09-14T10:00:00Z",
        temperature_c=28.0,
        humidity_pct=65.0,
        pressure_hpa=1012.0,
        source="IMD_AWS",
    )
    pipeline.process(live_input)

    # Ingest Demo Observation
    demo_input = CanonicalTelemetryInput(
        station_id=station_id,
        timestamp="2026-09-14T11:00:00Z",
        temperature_c=99.0,  # Corrupted data in demo
        humidity_pct=10.0,
        pressure_hpa=900.0,
        source="DEMO_SIMULATION",
        session_id="isolated_demo_run_123",
    )
    pipeline.process(demo_input)

    # Verify live history does not contain the demo record
    live_hist = station_history_live[station_id]
    assert len(live_hist) == 1
    assert live_hist[0]["temperature"] == 28.0

    # Verify demo history is isolated under session
    demo_hist = station_history_demo["isolated_demo_run_123"][station_id]
    assert len(demo_hist) == 1
    assert demo_hist[0]["temperature"] == 99.0


# =====================================================================
# 5. 7-Channel Structured Inference Response
# =====================================================================

def test_structured_inference_response_evidence_breakdown():
    """Validates structured response has all 7 evidence channels and calibrated metrics."""
    pipeline = get_canonical_pipeline()
    inp = CanonicalTelemetryInput(
        station_id="HYD_AWS_01",
        timestamp="2026-09-14T12:00:00Z",
        temperature_c=31.5,
        humidity_pct=62.0,
        pressure_hpa=1008.5,
        source="IMD_AWS",
    )
    res = pipeline.process(inp)

    assert isinstance(res, CanonicalInferenceResponse)
    assert res.station_id == "HYD_AWS_01"
    assert res.source == "IMD_AWS"

    # Evidence channels
    ev = res.evidence
    assert hasattr(ev, "data_quality")
    assert hasattr(ev, "statistical")
    assert hasattr(ev, "temporal")
    assert hasattr(ev, "ml")
    assert hasattr(ev, "multivariate")
    assert hasattr(ev, "spatial")
    assert hasattr(ev, "weather_context")

    # Non-probabilistic score bounds
    assert 0.0 <= res.anomaly_score <= 1.0
    assert 0.0 <= res.evidence_strength <= 1.0

    # Model metadata
    assert res.model.name != ""
    assert res.model.version != ""
    assert res.model.feature_schema_version != ""


# =====================================================================
# 6. Warmup State Handling
# =====================================================================

def test_warmup_state_for_new_station():
    """A new station with fewer than 24 observations reports WARMUP state for nominal readings."""
    pipeline = get_canonical_pipeline()
    stn_id = "NEW_WARMUP_STATION"

    inp = CanonicalTelemetryInput(
        station_id=stn_id,
        timestamp="2026-09-14T08:00:00Z",
        temperature_c=27.5,
        humidity_pct=65.0,
        pressure_hpa=1013.0,
        source="IMD_AWS",
    )
    res = pipeline.process(inp)

    assert res.warmup_state["is_warmup"] is True
    assert res.warmup_state["available_history"] < 24
    assert res.final_status == "WARMUP"
    assert res.is_anomaly is False


# =====================================================================
# 7. Classification States & Physical Validation
# =====================================================================

def test_data_quality_issue_corrupted_value():
    """Values outside physical atmospheric bounds trigger DATA_QUALITY_ISSUE & CRITICAL."""
    pipeline = get_canonical_pipeline()
    inp = CanonicalTelemetryInput(
        station_id="HYD_AWS_01",
        timestamp="2026-09-14T14:00:00Z",
        temperature_c=999.0,  # Unphysical
        humidity_pct=50.0,
        pressure_hpa=1010.0,
        source="IMD_AWS",
    )
    res = pipeline.process(inp)
    assert res.final_status == "DATA_QUALITY_ISSUE"
    assert res.anomaly_type in ("CORRUPTED_DATA", "DATA_QUALITY_ANOMALY")
    assert res.severity == "CRITICAL"

    assert res.is_anomaly is True


def test_communication_failure_missing_payload():
    """Payload with all primary parameters missing triggers COMMUNICATION_FAILURE."""
    pipeline = get_canonical_pipeline()
    inp = CanonicalTelemetryInput(
        station_id="HYD_AWS_01",
        timestamp="2026-09-14T15:00:00Z",
        temperature_c=None,
        humidity_pct=None,
        pressure_hpa=None,
        source="IMD_AWS",
    )
    res = pipeline.process(inp)
    assert res.final_status == "COMMUNICATION_FAILURE"
    assert res.anomaly_type == "COMMUNICATION_FAILURE"
    assert res.severity == "HIGH"
    assert res.is_anomaly is True


# =====================================================================
# 8. Anomaly Lifecycle & Alert Deduplication
# =====================================================================

def test_alert_deduplication_and_lifecycle():
    """Consecutive readings of continuing fault update single incident without spamming duplicates."""
    mgr = IncidentManager()
    station = "DEDUP_AWS_01"

    # 1. First anomalous observation
    inc1 = mgr.process_observation_incident(
        station_id=station,
        timestamp_str="2026-09-14T10:00:00Z",
        is_anomaly=True,
        anomaly_type="TEMPERATURE_SPIKE",
        severity="HIGH",
    )
    assert inc1 is not None
    assert inc1.status == "DETECTED"
    assert inc1.occurrence_count == 1
    assert inc1.duration_seconds == 0.0
    first_id = inc1.incident_id

    # 2. Second consecutive anomalous observation 10 minutes later
    inc2 = mgr.process_observation_incident(
        station_id=station,
        timestamp_str="2026-09-14T10:10:00Z",
        is_anomaly=True,
        anomaly_type="TEMPERATURE_SPIKE",
        severity="HIGH",
    )
    assert inc2 is not None
    # Must preserve same persistent incident ID
    assert inc2.incident_id == first_id
    assert inc2.occurrence_count == 2
    assert inc2.duration_seconds == 600.0

    # 3. Acknowledge the incident
    updated = mgr.update_lifecycle_status(first_id, "ACKNOWLEDGED")
    assert updated is not None
    assert updated["status"] == "ACKNOWLEDGED"

    # 4. Move to Investigating
    updated2 = mgr.update_lifecycle_status(first_id, "INVESTIGATING")
    assert updated2 is not None
    assert updated2["status"] == "INVESTIGATING"

    # 5. Consecutive normal observations auto-resolve
    mgr.process_observation_incident(station, "2026-09-14T10:20:00Z", is_anomaly=False, anomaly_type="NONE", severity="LOW")
    mgr.process_observation_incident(station, "2026-09-14T10:30:00Z", is_anomaly=False, anomaly_type="NONE", severity="LOW")

    # Verify incident moved to resolved history
    history = mgr.list_incidents(status="RESOLVED", station_id=station)
    assert len(history) >= 1
    assert history[0]["incident_id"] == first_id


# =====================================================================
# 9. Persistent History Across Server Restart
# =====================================================================

def test_persistence_history_hydration_on_restart():
    """In-memory cache recovers historical observations from persistent storage after server restart."""
    pipeline = get_canonical_pipeline()
    station = f"RESTART_STN_{uuid.uuid4().hex[:8]}"

    # Ingest 3 readings
    for h in range(3):
        ts = f"2026-09-14T0{h}:00:00Z"
        inp = CanonicalTelemetryInput(
            station_id=station,
            timestamp=ts,
            temperature_c=25.0 + h,
            humidity_pct=60.0,
            pressure_hpa=1012.0,
            source="IMD_AWS",
        )
        pipeline.process(inp)

    assert len(station_history_live[station]) == 3

    # Simulate Server Restart by clearing in-memory cache
    station_history_live.clear()
    assert len(station_history_live[station]) == 0

    # Hydrate history from persistent database service
    hydrated = pipeline.hydrate_history_if_needed(station, is_live=True)
    assert len(hydrated) >= 3
    assert hydrated[-1]["temperature"] == 27.0


# =====================================================================
# 10. FastAPI HTTP Endpoints Verification
# =====================================================================

def test_http_post_telemetry_canonical(client):
    """POST /api/telemetry returns full CanonicalInferenceResponse."""
    payload = {
        "station_id": "HYD_AWS_01",
        "timestamp": "2026-09-14T16:00:00Z",
        "temp": 33.2,
        "hum": 58.0,
        "baro": 1009.1,
        "source": "IMD_AWS",
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == "HYD_AWS_01"
    assert data["source"] == "IMD_AWS"
    assert "final_status" in data
    assert "anomaly_score" in data
    assert "evidence" in data
    assert "data_quality" in data["evidence"]
    assert "ml" in data["evidence"]
    assert "model" in data


def test_http_post_telemetry_rejects_invalid_source(client):
    """POST /api/telemetry rejects invalid source identifiers."""
    payload = {
        "station_id": "HYD_AWS_01",
        "timestamp": "2026-09-14T16:00:00Z",
        "temp": 30.0,
        "source": "INVALID_UNAPPROVED_FEED",
    }
    response = client.post("/api/telemetry", json=payload)
    assert response.status_code == 422


def test_http_incidents_lifecycle_endpoints(client):
    """GET /api/incidents and PATCH /api/incidents/{id}/status endpoints."""
    # First generate an incident via telemetry
    payload = {
        "station_id": "HYD_AWS_01",
        "timestamp": "2026-09-14T18:00:00Z",
        "temperature_c": 999.0,  # Corrupted data fault
        "humidity_pct": 50.0,
        "pressure_hpa": 1010.0,
        "source": "IMD_AWS",
    }
    res = client.post("/api/telemetry", json=payload)
    assert res.status_code == 200
    data = res.json()
    incident_id = data["incident"]["incident_id"]

    # Query active incidents
    get_res = client.get("/api/incidents")
    assert get_res.status_code == 200
    inc_list = get_res.json()["incidents"]
    assert any(i["incident_id"] == incident_id for i in inc_list)

    # Patch incident status to ACKNOWLEDGED
    patch_res = client.patch(f"/api/incidents/{incident_id}/status?status=ACKNOWLEDGED")
    assert patch_res.status_code == 200
    assert patch_res.json()["incident"]["status"] == "ACKNOWLEDGED"


# =====================================================================
# 11. Source-Label Integrity & Station Identity Tests
# =====================================================================

def test_station_inventory_endpoint_identifies_simulated_stations(client):
    """GET /api/stations strictly distinguishes real IMD AWS stations from simulated/indicative ones."""
    res = client.get("/api/stations")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["real_imd_count"] == 3

    stn_map = {s["station_id"]: s for s in data["stations"]}

    # Real IMD AWS stations
    for real_id in ("43189", "43150", "43245"):
        assert real_id in stn_map
        assert stn_map[real_id]["is_simulated"] is False
        assert stn_map[real_id]["station_type"] == "IMD_AWS_REFERENCE"

    # Indicative simulated stations
    for sim_id in ("AWS-101", "AWS-102", "HYD_AWS_01", "BLR_AWS_02", "DEL_AWS_03"):
        assert sim_id in stn_map
        assert stn_map[sim_id]["is_simulated"] is True
        assert stn_map[sim_id]["station_type"] == "SIMULATED_INDICATIVE"


def test_live_imd_observation_simulated_station_labeled_accurately(client):
    """Simulated station requested from live endpoint must never be labeled as IMD_AWS."""
    res = client.get("/api/live/imd-observation?station_id=AWS-101&run_pipeline=false")
    assert res.status_code == 200
    data = res.json()
    assert data["station_id"] == "AWS-101"
    assert data["is_simulated"] is True
    assert data["imd_fetch_status"] == "FAILED"
    assert "simulated/indicative" in data["imd_failure_reason"]
    # Source must be OPEN_METEO_CONTEXT or OFFLINE_PREVIEW, NEVER IMD_AWS
    assert data["source"] in ("OPEN_METEO_CONTEXT", "OFFLINE_PREVIEW")
    assert data["source"] != "IMD_AWS"


def test_live_imd_observation_records_failure_on_real_station_error(client, monkeypatch):
    """When IMD API fails for a real station, failure is recorded and source is not falsely labeled as IMD_AWS."""
    import requests

    def mock_get_fail(*args, **kwargs):
        raise requests.RequestException("IMD AWS upstream gateway timeout (simulated)")

    monkeypatch.setattr(requests, "get", mock_get_fail)

    res = client.get("/api/live/imd-observation?station_id=43189&run_pipeline=false")
    assert res.status_code == 200
    data = res.json()
    assert data["station_id"] == "43189"
    assert data["is_simulated"] is False
    assert data["imd_fetch_status"] == "FAILED"
    assert "IMD AWS upstream gateway timeout" in data["imd_failure_reason"]
    assert data["source"] in ("OPEN_METEO_CONTEXT", "OFFLINE_PREVIEW")
    assert data["source"] != "IMD_AWS"


def test_process_pipeline_prevents_simulated_station_as_real_imd(client):
    """Simulated station sent to /api/pipeline/process with source=LIVE is not mislabeled as IMD_AWS."""
    payload = {
        "station_id": "AWS-101",  # Simulated
        "observation": {
            "temperature": 27.5,
            "humidity": 65.0,
            "pressure": 1010.0,
        },
        "source": "LIVE",
    }
    res = client.post("/api/pipeline/process", json=payload)
    assert res.status_code == 200
    data = res.json()
    # Must NOT be labeled as IMD_AWS for simulated station AWS-101
    assert data["source"] != "IMD_AWS"
    assert data["source"] in ("OPEN_METEO_CONTEXT", "DEMO_SIMULATION")


def test_persistence_preserves_all_standardized_sources():
    """All 4 standardized sources survive through local SQLite persistence."""
    from app.services.database_service import save_sensor_reading, get_latest_readings

    for src in ("IMD_AWS", "DEMO_SIMULATION", "OPEN_METEO_CONTEXT", "OFFLINE_PREVIEW"):
        now_ts = datetime.now(timezone.utc).isoformat()
        rec = save_sensor_reading(
            station_id="43189",
            timestamp=now_ts,
            temperature_c=29.0,
            humidity_pct=65.0,
            pressure_hpa=1010.0,
            source=src,
        )
        assert rec is not None
        assert rec["source"] == src

