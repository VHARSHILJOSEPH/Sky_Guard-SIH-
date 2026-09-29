"""
SkyGuard AI — Persistent Database Layer Test Suite
Validates Supabase and local SQLite persistence for:
1. stations inventory and metadata
2. sensor_readings telemetry and data quality status
3. data isolation between DEMO_SIMULATION and operational telemetry
4. anomalies with 7-channel evidence and lifecycle status tracking
5. station_health availability and observation telemetry counters
6. model_versions governance, thresholds, and benchmark metrics
"""

from datetime import datetime, timezone
import json
import pytest

# pyrefly: ignore [missing-import]
from app.services.database_service import (
    check_database_health,
    save_sensor_reading,
    get_latest_readings,
    get_station_history_chronological,
    save_anomaly,
    update_anomaly_status,
    get_recent_anomalies,
    save_station_health,
    get_station_health,
    save_model_version,
    get_model_versions,
    seed_stations_inventory,
    get_stations,
)


def test_schema_compatibility_and_health():
    """Verify database connectivity and health check response."""
    health = check_database_health()
    assert isinstance(health, dict)
    assert "status" in health
    assert health["status"] in ("healthy", "unhealthy")
    assert health["provider"] == "supabase"


def test_insert_and_retrieve_sensor_reading():
    """Verify sensor reading insertion with complete telemetry attributes."""
    ts = datetime.now(timezone.utc).isoformat()
    reading = save_sensor_reading(
        station_id="43189",
        timestamp=ts,
        temperature_c=31.5,
        humidity_pct=62.0,
        pressure_hpa=1010.5,
        wind_speed_ms=4.2,
        wind_direction_deg=180.0,
        rainfall_mm=0.0,
        source="IMD_AWS",
        session_id=None,
        quality_status="VALID",
        raw_payload={"temperature": 31.5, "humidity": 62.0, "pressure": 1010.5},
    )

    assert reading is not None
    assert reading["station_id"] == "43189"
    assert reading["temperature_c"] == 31.5
    assert reading["humidity_pct"] == 62.0
    assert reading["pressure_hpa"] == 1010.5
    assert reading["source"] == "IMD_AWS"
    assert reading["quality_status"] == "VALID"

    # Retrieve and verify presence
    latest = get_latest_readings(limit=5, station_id="43189", include_demo=False)
    assert len(latest) > 0
    match = next((r for r in latest if r.get("timestamp") == ts), None)
    assert match is not None
    assert match.get("temperature_c") == 31.5


def test_data_isolation_demo_vs_live():
    """
    CRITICAL REQUIREMENT:
    Verify DEMO_SIMULATION records never contaminate operational live queries.
    """
    session_id = f"test_isolation_{int(datetime.now(timezone.utc).timestamp())}"
    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Insert an operational reading
    live_rec = save_sensor_reading(
        station_id="43150",
        timestamp=now_ts,
        temperature_c=29.5,
        humidity_pct=75.0,
        pressure_hpa=1011.0,
        source="IMD_AWS",
    )
    assert live_rec["source"] == "IMD_AWS"

    # 2. Insert a synthetic demo replay observation
    demo_rec = save_sensor_reading(
        station_id="43150",
        timestamp=now_ts,
        temperature_c=99.9,
        humidity_pct=99.9,
        pressure_hpa=999.9,
        source="DEMO_SIMULATION",
        session_id=session_id,
        quality_status="SIMULATED",
    )
    assert demo_rec["source"] == "DEMO_SIMULATION"
    assert demo_rec["session_id"] == session_id

    # 3. Query operational history (default include_demo=False)
    operational_history = get_latest_readings(limit=20, station_id="43150", include_demo=False)
    assert len(operational_history) > 0
    # Assert NO DEMO_SIMULATION record appears in operational history
    for item in operational_history:
        assert item.get("source") != "DEMO_SIMULATION", (
            f"Isolation breach! Found DEMO_SIMULATION reading in operational query: {item}"
        )

    # 4. Query chronological history
    chronological_ops = get_station_history_chronological(station_id="43150", limit=20, include_demo=False)
    for obs in chronological_ops:
        assert obs.get("temperature") != 99.9, "Isolation breach: simulated spike leaked into chronological history!"

    # 5. Query demo session explicitly
    demo_query = get_latest_readings(limit=10, session_id=session_id, include_demo=True)
    assert len(demo_query) > 0
    assert demo_query[0]["session_id"] == session_id
    assert demo_query[0]["source"] == "DEMO_SIMULATION"


def test_insert_and_retrieve_anomaly_with_evidence():
    """Verify anomaly record persistence with 7-channel evidence and ML provenance."""
    ts = datetime.now(timezone.utc).isoformat()
    evidence_payload = {
        "ml": {"score": 0.88, "threshold": 0.12, "model": "LOF_BME280", "ml_anomaly": True},
        "temporal": {"spike_score": 0.85, "drift_score": 0.0, "frozen_score": 0.0},
        "statistical": {"score": 0.75, "baseline_deviation": 11.5},
        "multivariate": {"score": 0.60},
        "spatial": {"score": 0.90, "peer_disagreement": True},
        "weather_context": {"score": 0.10},
        "data_quality": {"score": 1.0, "valid": True},
    }

    anomaly = save_anomaly(
        station_id="43245",
        timestamp=ts,
        anomaly_type="TEMPERATURE_SPIKE",
        final_status="ANOMALY",
        severity="HIGH",
        affected_parameter="temperature",
        observed_value=45.2,
        expected_value=33.1,
        anomaly_score=0.88,
        confidence=92.5,
        explanation="Sudden +12.1°C excursion unsupported by peer stations or synoptic forecast.",
        evidence=evidence_payload,
        model_name="SkyGuard-LOF-BME280",
        model_version="LOF_BME280_v1.0",
        status="DETECTED",
    )

    assert anomaly is not None
    assert anomaly["station_id"] == "43245"
    assert anomaly["anomaly_type"] == "TEMPERATURE_SPIKE"
    assert anomaly["status"] == "DETECTED"
    assert anomaly["severity"] == "HIGH"

    # Verify retrieval
    recent = get_recent_anomalies(limit=10, station_id="43245")
    assert len(recent) > 0
    match = next((a for a in recent if a.get("timestamp") == ts), None)
    assert match is not None
    assert match["anomaly_type"] == "TEMPERATURE_SPIKE"


def test_update_anomaly_lifecycle():
    """Verify incident lifecycle progression from DETECTED to RESOLVED."""
    ts = datetime.now(timezone.utc).isoformat()
    anomaly = save_anomaly(
        station_id="43189",
        timestamp=ts,
        anomaly_type="FROZEN_SENSOR",
        final_status="ANOMALY",
        severity="MEDIUM",
        status="DETECTED",
        explanation="Zero variance detected over consecutive hours.",
    )
    anom_id = anomaly.get("id")
    assert anom_id is not None

    # 1. Transition to ACKNOWLEDGED
    ack = update_anomaly_status(anomaly_id=anom_id, status="ACKNOWLEDGED")
    assert ack["status"] == "ACKNOWLEDGED"

    # 2. Transition to RESOLVED
    res_time = datetime.now(timezone.utc).isoformat()
    resolved = update_anomaly_status(anomaly_id=anom_id, status="RESOLVED", resolved_at=res_time)
    assert resolved["status"] == "RESOLVED"
    assert resolved.get("resolved_at") is not None


def test_station_health_persistence():
    """Verify station health availability and observation counters."""
    ts = datetime.now(timezone.utc).isoformat()
    health = save_station_health(
        station_id="43189",
        health_score=95,
        status="HEALTHY",
        issues=["minor_packet_jitter"],
        timestamp=ts,
        availability=99.2,
        expected_observations=144,
        received_observations=143,
        missing_observations=1,
        anomaly_frequency=0.02,
        false_alarm_metrics={"false_alarms": 0, "corroboration_rate": 0.98},
        last_seen=ts,
    )

    assert health is not None
    assert health["station_id"] == "43189"
    assert health["health_score"] == 95
    assert health["status"] == "HEALTHY"

    # Retrieve and verify
    records = get_station_health(station_id="43189", limit=5)
    assert len(records) > 0
    assert records[0]["health_score"] == 95


def test_model_version_persistence_and_governance():
    """Verify ML model version governance record persistence."""
    model_data = {
        "model_name": "SkyGuard-LOF-BME280",
        "model_version": "LOF_BME280_v1.0",
        "algorithm": "LocalOutlierFactor",
        "dataset_version": "benchmark_v1.0_clean",
        "feature_schema_version": "features_v1.0_temp_hum_pres_wind",
        "threshold": 1.50,
        "training_metadata": {
            "contamination": 0.05,
            "n_neighbors": 20,
            "metric": "minkowski",
            "training_split": "clean_train_only",
        },
        "evaluation_metrics": {
            "precision": 0.952,
            "recall": 0.941,
            "f1": 0.946,
            "specificity": 0.988,
            "balanced_accuracy": 0.965,
            "mcc": 0.912,
        },
        "artifact_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }

    saved = save_model_version(model_data)
    assert saved is not None
    assert saved.get("model_version") == "LOF_BME280_v1.0"

    all_models = get_model_versions(limit=10)
    assert len(all_models) > 0
    found = next((m for m in all_models if m.get("model_version") == "LOF_BME280_v1.0"), None)
    assert found is not None
    assert found.get("algorithm") == "LocalOutlierFactor"


def test_stations_inventory_seeding_and_retrieval():
    """Verify stations inventory seeding and distinguishes simulated vs reference."""
    inventory = [
        {
            "station_id": "43189",
            "name": "Vijayawada (AWS014)",
            "latitude": 16.5062,
            "longitude": 80.6480,
            "state": "Andhra Pradesh",
            "source_type": "IMD_AWS_REFERENCE",
            "is_simulated": False,
            "status": "ACTIVE",
        },
        {
            "station_id": "HYD_AWS_01",
            "name": "Hyderabad Demo Station",
            "latitude": 17.3850,
            "longitude": 78.4867,
            "state": "Telangana",
            "source_type": "SIMULATED_INDICATIVE",
            "is_simulated": True,
            "status": "ACTIVE",
        },
    ]

    count = seed_stations_inventory(inventory)
    assert count >= 2

    stations = get_stations()
    assert len(stations) >= 2

    # Check that is_simulated flag is accurate
    ref = next((s for s in stations if s["station_id"] == "43189"), None)
    sim = next((s for s in stations if s["station_id"] == "HYD_AWS_01"), None)
    assert ref is not None
    assert sim is not None
    assert ref.get("is_simulated") in (False, 0)
    assert sim.get("is_simulated") in (True, 1)
