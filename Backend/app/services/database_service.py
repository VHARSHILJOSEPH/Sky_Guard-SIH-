"""
SkyGuard AI — Supabase & Persistent Database Service
Manages persistence for:
1. stations (inventory and metadata)
2. sensor_readings (operational and session telemetry)
3. anomalies (lifecycle-tracked incident records with multi-channel evidence)
4. station_health (availability, health score, observation counters)
5. model_versions (ML model governance, thresholds, and benchmark metrics)

Implements:
- Supabase cloud persistence with service-role security
- Local SQLite persistence for offline resilience and restart survival
- Schema adaptability (graceful fallback if cloud migrations are pending)
- Strict data isolation between DEMO_SIMULATION and operational telemetry
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Optional

from app.database.supabase_client import supabase, SUPABASE_URL, BACKEND_KEY

logger = logging.getLogger("skyguard.database")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH = DATA_DIR / "skyguard_local_history.db"

_REMOTE_SCHEMA_CACHE: dict[str, set[str]] = {}


def _init_sqlite_db() -> None:
    """Initialize local SQLite persistence tables and ensure all columns exist."""
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()

            # 1. stations_local
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stations_local (
                    station_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    state TEXT,
                    district TEXT,
                    source_type TEXT NOT NULL DEFAULT 'IMD_AWS_REFERENCE',
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    is_simulated INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 2. sensor_readings_local
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings_local (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    temperature_c REAL,
                    humidity_pct REAL,
                    pressure_hpa REAL,
                    wind_speed_ms REAL,
                    wind_direction_deg REAL,
                    rainfall_mm REAL,
                    source TEXT DEFAULT 'IMD_AWS',
                    session_id TEXT,
                    quality_status TEXT DEFAULT 'VALID',
                    raw_payload TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            cursor.execute("PRAGMA table_info(sensor_readings_local)")
            existing_sr_cols = {col[1] for col in cursor.fetchall()}
            for col_name, col_type in [
                ("wind_direction_deg", "REAL"),
                ("session_id", "TEXT"),
                ("quality_status", "TEXT DEFAULT 'VALID'"),
                ("raw_payload", "TEXT"),
            ]:
                if col_name not in existing_sr_cols:
                    cursor.execute(f"ALTER TABLE sensor_readings_local ADD COLUMN {col_name} {col_type}")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_stn_time ON sensor_readings_local(station_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sr_source ON sensor_readings_local(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sr_session ON sensor_readings_local(session_id)")

            # 3. anomalies_local
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies_local (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reading_id TEXT,
                    station_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    final_status TEXT NOT NULL DEFAULT 'ANOMALY',
                    severity TEXT NOT NULL DEFAULT 'LOW',
                    affected_parameter TEXT,
                    observed_value REAL,
                    expected_value REAL,
                    anomaly_score REAL NOT NULL DEFAULT 0.0,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    evidence TEXT,
                    explanation TEXT,
                    model_name TEXT NOT NULL DEFAULT 'SkyGuard-LOF-BME280',
                    model_version TEXT NOT NULL DEFAULT 'v1.0',
                    status TEXT NOT NULL DEFAULT 'DETECTED',
                    first_detected TEXT,
                    last_detected TEXT,
                    occurrence_count INTEGER NOT NULL DEFAULT 1,
                    session_id TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
            """)

            cursor.execute("PRAGMA table_info(anomalies_local)")
            existing_anom_cols = {col[1] for col in cursor.fetchall()}
            for col_name, col_type in [
                ("final_status", "TEXT NOT NULL DEFAULT 'ANOMALY'"),
                ("evidence", "TEXT"),
                ("model_name", "TEXT NOT NULL DEFAULT 'SkyGuard-LOF-BME280'"),
                ("model_version", "TEXT NOT NULL DEFAULT 'v1.0'"),
                ("status", "TEXT NOT NULL DEFAULT 'DETECTED'"),
                ("first_detected", "TEXT"),
                ("last_detected", "TEXT"),
                ("occurrence_count", "INTEGER NOT NULL DEFAULT 1"),
                ("session_id", "TEXT"),
                ("resolved_at", "TEXT"),
            ]:
                if col_name not in existing_anom_cols:
                    cursor.execute(f"ALTER TABLE anomalies_local ADD COLUMN {col_name} {col_type}")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anom_stn_time ON anomalies_local(station_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anom_status ON anomalies_local(status)")

            # 4. station_health_local
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS station_health_local (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    availability REAL NOT NULL DEFAULT 100.0,
                    expected_observations INTEGER NOT NULL DEFAULT 0,
                    received_observations INTEGER NOT NULL DEFAULT 0,
                    missing_observations INTEGER NOT NULL DEFAULT 0,
                    anomaly_frequency REAL NOT NULL DEFAULT 0.0,
                    false_alarm_metrics TEXT,
                    health_score INTEGER NOT NULL DEFAULT 100,
                    status TEXT NOT NULL DEFAULT 'HEALTHY',
                    issues TEXT,
                    last_seen TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            cursor.execute("PRAGMA table_info(station_health_local)")
            existing_health_cols = {col[1] for col in cursor.fetchall()}
            for col_name, col_type in [
                ("availability", "REAL NOT NULL DEFAULT 100.0"),
                ("expected_observations", "INTEGER NOT NULL DEFAULT 0"),
                ("received_observations", "INTEGER NOT NULL DEFAULT 0"),
                ("missing_observations", "INTEGER NOT NULL DEFAULT 0"),
                ("anomaly_frequency", "REAL NOT NULL DEFAULT 0.0"),
                ("false_alarm_metrics", "TEXT"),
                ("last_seen", "TEXT"),
                ("updated_at", "TEXT"),
            ]:
                if col_name not in existing_health_cols:
                    cursor.execute(f"ALTER TABLE station_health_local ADD COLUMN {col_name} {col_type}")

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_health_stn_time ON station_health_local(station_id, timestamp)")

            # 5. model_versions_local
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions_local (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL UNIQUE,
                    algorithm TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    feature_schema_version TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    training_metadata TEXT,
                    evaluation_metrics TEXT,
                    artifact_hash TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            conn.commit()
    except Exception as exc:
        logger.warning(f"[DatabaseService] Local SQLite init notice: {exc}")


_init_sqlite_db()


def _get_remote_columns(table_name: str) -> set[str]:
    """Retrieve and cache column names supported by the remote Supabase table."""
    global _REMOTE_SCHEMA_CACHE
    if table_name in _REMOTE_SCHEMA_CACHE:
        return _REMOTE_SCHEMA_CACHE[table_name]

    import requests
    try:
        headers = {"apikey": BACKEND_KEY, "Authorization": f"Bearer {BACKEND_KEY}"}
        res = requests.get(f"{SUPABASE_URL}/rest/v1/", headers=headers, timeout=5)
        if res.ok:
            definitions = res.json().get("definitions", {})
            if table_name in definitions:
                cols = set(definitions[table_name].get("properties", {}).keys())
                _REMOTE_SCHEMA_CACHE[table_name] = cols
                return cols
    except Exception as exc:
        logger.debug(f"[DatabaseService] Could not inspect remote schema for {table_name}: {exc}")

    return set()


def _sanitize_float(val: Any) -> Optional[float]:
    """Sanitize float values, converting NaN, Inf, and invalid types to None."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def _format_timestamp(ts: Any) -> str:
    """Ensure timestamp is a valid ISO 8601 string in UTC."""
    if not ts:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    s = str(ts).strip()
    if not s or s.lower() in ("none", "nat", "nan", "null"):
        return datetime.now(timezone.utc).isoformat()
    return s


def check_database_health() -> dict[str, str]:
    """Check connectivity to Supabase database."""
    try:
        supabase.table("stations").select("station_id").limit(1).execute()
        return {
            "database": "connected",
            "provider": "supabase",
            "status": "healthy",
        }
    except Exception as exc:
        logger.error(f"[DatabaseService] Supabase health check failed: {exc}")
        return {
            "database": "disconnected",
            "provider": "supabase",
            "status": "unhealthy",
        }


# =====================================================================
# 1. SENSOR READINGS PERSISTENCE
# =====================================================================

def save_sensor_reading(
    station_id: str,
    timestamp: Any,
    temperature_c: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    pressure_hpa: Optional[float] = None,
    wind_speed_ms: Optional[float] = None,
    wind_direction_deg: Optional[float] = None,
    rainfall_mm: Optional[float] = None,
    source: str = "IMD_AWS",
    session_id: Optional[str] = None,
    quality_status: str = "VALID",
    raw_payload: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Persist a normalized sensor reading to local SQLite store and Supabase."""
    clean_source = str(source or "").strip().upper()
    if clean_source in ("LIVE", "LIVE_AWS", "IMD", "IMD_AWS"):
        clean_source = "IMD_AWS"
    elif clean_source in ("DEMO", "SIMULATION", "DEMO_REPLAY", "SYNTHETIC", "DEMO_SIMULATION"):
        clean_source = "DEMO_SIMULATION"
    elif clean_source in ("OPEN_METEO", "FORECAST", "OPENMETEO", "OPEN_METEO_CONTEXT"):
        clean_source = "OPEN_METEO_CONTEXT"
    elif clean_source in ("OFFLINE", "PREVIEW", "OFFLINE_PREVIEW"):
        clean_source = "OFFLINE_PREVIEW"
    elif clean_source not in ("IMD_AWS", "DEMO_SIMULATION", "OPEN_METEO_CONTEXT", "OFFLINE_PREVIEW"):
        clean_source = "OFFLINE_PREVIEW"

    ts_formatted = _format_timestamp(timestamp)
    t_val = _sanitize_float(temperature_c)
    h_val = _sanitize_float(humidity_pct)
    p_val = _sanitize_float(pressure_hpa)
    w_val = _sanitize_float(wind_speed_ms)
    wd_val = _sanitize_float(wind_direction_deg)
    r_val = _sanitize_float(rainfall_mm)
    now_iso = datetime.now(timezone.utc).isoformat()
    raw_payload_json = json.dumps(raw_payload) if raw_payload else None

    local_record: dict[str, Any] = {
        "station_id": str(station_id).strip(),
        "timestamp": ts_formatted,
        "temperature_c": t_val,
        "humidity_pct": h_val,
        "pressure_hpa": p_val,
        "wind_speed_ms": w_val,
        "wind_direction_deg": wd_val,
        "rainfall_mm": r_val,
        "source": clean_source,
        "session_id": session_id,
        "quality_status": quality_status,
        "raw_payload": raw_payload_json,
        "created_at": now_iso,
    }

    # 1. Persist to SQLite
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO sensor_readings_local (
                    station_id, timestamp, temperature_c, humidity_pct, pressure_hpa,
                    wind_speed_ms, wind_direction_deg, rainfall_mm, source, session_id,
                    quality_status, raw_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    local_record["station_id"],
                    local_record["timestamp"],
                    local_record["temperature_c"],
                    local_record["humidity_pct"],
                    local_record["pressure_hpa"],
                    local_record["wind_speed_ms"],
                    local_record["wind_direction_deg"],
                    local_record["rainfall_mm"],
                    local_record["source"],
                    local_record["session_id"],
                    local_record["quality_status"],
                    local_record["raw_payload"],
                    local_record["created_at"],
                ),
            )
            local_record["id"] = cursor.lastrowid
            conn.commit()
    except Exception as sq_exc:
        logger.warning(f"[DatabaseService] SQLite insert notice: {sq_exc}")

    # 2. Persist to Supabase
    try:
        payload = {
            "station_id": local_record["station_id"],
            "timestamp": local_record["timestamp"],
            "temperature_c": local_record["temperature_c"],
            "humidity_pct": local_record["humidity_pct"],
            "pressure_hpa": local_record["pressure_hpa"],
            "wind_speed_ms": local_record["wind_speed_ms"],
            "wind_direction_deg": local_record["wind_direction_deg"],
            "rainfall_mm": local_record["rainfall_mm"],
            "source": local_record["source"],
            "session_id": local_record["session_id"],
            "quality_status": local_record["quality_status"],
            "raw_payload": raw_payload,
        }
        remote_cols = _get_remote_columns("sensor_readings")
        if remote_cols:
            payload = {k: v for k, v in payload.items() if k in remote_cols}

        res = supabase.table("sensor_readings").insert(payload).execute()
        if res.data and len(res.data) > 0:
            merged = dict(local_record)
            merged.update(res.data[0])
            return merged
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase insert notice: {exc}")

    return local_record


def get_latest_readings(
    limit: int = 10,
    station_id: Optional[str] = None,
    source: Optional[str] = None,
    include_demo: bool = False,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve recent sensor readings, checking Supabase first and falling back to SQLite."""
    # 1. Query Supabase
    try:
        query = supabase.table("sensor_readings").select("*").order("timestamp", desc=True).limit(limit)
        if station_id:
            query = query.eq("station_id", station_id)
        if source:
            query = query.eq("source", source)
        elif not include_demo:
            query = query.neq("source", "DEMO_SIMULATION")
        if session_id:
            query = query.eq("session_id", session_id)

        res = query.execute()
        if res.data and len(res.data) > 0:
            return res.data
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase query notice: {exc}")

    # 2. Fallback to SQLite
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            clauses = []
            params: list[Any] = []

            if station_id:
                clauses.append("station_id = ?")
                params.append(station_id)

            if source:
                clauses.append("source = ?")
                params.append(source)
            elif not include_demo:
                clauses.append("source != 'DEMO_SIMULATION'")

            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)

            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = f"SELECT * FROM sensor_readings_local {where_sql} ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, tuple(params))
            rows = [dict(r) for r in cursor.fetchall()]
            return rows
    except Exception as sq_exc:
        logger.error(f"[DatabaseService] SQLite fetch error: {sq_exc}")
        return []


def get_station_readings(
    station_id: str,
    limit: int = 50,
    include_demo: bool = False,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve historical sensor readings for a station in reverse chronological order."""
    return get_latest_readings(
        limit=limit,
        station_id=station_id,
        include_demo=include_demo,
        session_id=session_id,
    )


def get_station_history_chronological(
    station_id: str,
    limit: int = 50,
    include_demo: bool = False,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve historical observations for a station in chronological (ascending) order."""
    reverse_readings = get_station_readings(
        station_id=station_id,
        limit=limit,
        include_demo=include_demo,
        session_id=session_id,
    )
    if not reverse_readings:
        return []

    formatted = []
    for r in reversed(reverse_readings):
        formatted.append({
            "station_id": r.get("station_id"),
            "timestamp": r.get("timestamp"),
            "temperature": r.get("temperature_c") if "temperature_c" in r else r.get("temperature"),
            "humidity": r.get("humidity_pct") if "humidity_pct" in r else r.get("humidity"),
            "pressure": r.get("pressure_hpa") if "pressure_hpa" in r else r.get("pressure"),
            "wind_speed": r.get("wind_speed_ms") if "wind_speed_ms" in r else r.get("wind_speed", 0.0),
            "wind_direction": r.get("wind_direction_deg") if "wind_direction_deg" in r else r.get("wind_direction", 0.0),
            "rainfall": r.get("rainfall_mm") if "rainfall_mm" in r else r.get("rainfall", 0.0),
        })
    return formatted


# =====================================================================
# 2. ANOMALY & INCIDENT LIFECYCLE PERSISTENCE
# =====================================================================

def save_anomaly(
    station_id: str,
    timestamp: Any,
    anomaly_type: str,
    reading_id: Optional[Any] = None,
    severity: str = "LOW",
    affected_parameter: Optional[str] = None,
    observed_value: Any = None,
    expected_value: Any = None,
    anomaly_score: float = 0.0,
    confidence: float = 0.0,
    explanation: Optional[str] = "",
    final_status: str = "ANOMALY",
    evidence: Optional[dict[str, Any]] = None,
    model_name: str = "SkyGuard-LOF-BME280",
    model_version: str = "v1.0",
    status: str = "DETECTED",
    first_detected: Optional[Any] = None,
    last_detected: Optional[Any] = None,
    occurrence_count: int = 1,
    session_id: Optional[str] = None,
    resolved_at: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Persist an identified anomaly with multi-channel evidence and lifecycle status."""
    ts_formatted = _format_timestamp(timestamp)
    first_det_iso = _format_timestamp(first_detected or timestamp)
    last_det_iso = _format_timestamp(last_detected or timestamp)
    resolved_iso = _format_timestamp(resolved_at) if resolved_at else None
    now_iso = datetime.now(timezone.utc).isoformat()
    evidence_dict = evidence or {}
    evidence_json = json.dumps(evidence_dict)

    local_record: dict[str, Any] = {
        "reading_id": str(reading_id) if reading_id else None,
        "station_id": str(station_id).strip(),
        "timestamp": ts_formatted,
        "anomaly_type": str(anomaly_type).strip(),
        "final_status": str(final_status).strip().upper(),
        "severity": str(severity).strip().upper(),
        "affected_parameter": str(affected_parameter).strip() if affected_parameter else "telemetry",
        "observed_value": _sanitize_float(observed_value),
        "expected_value": _sanitize_float(expected_value),
        "anomaly_score": float(anomaly_score) if anomaly_score is not None else 0.0,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "evidence": evidence_json,
        "explanation": str(explanation).strip() if explanation else "",
        "model_name": str(model_name).strip(),
        "model_version": str(model_version).strip(),
        "status": str(status).strip().upper(),
        "first_detected": first_det_iso,
        "last_detected": last_det_iso,
        "occurrence_count": int(occurrence_count),
        "session_id": session_id,
        "created_at": now_iso,
        "resolved_at": resolved_iso,
    }

    # 1. SQLite persistence
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO anomalies_local (
                    reading_id, station_id, timestamp, anomaly_type, final_status,
                    severity, affected_parameter, observed_value, expected_value,
                    anomaly_score, confidence, evidence, explanation, model_name,
                    model_version, status, first_detected, last_detected,
                    occurrence_count, session_id, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    local_record["reading_id"],
                    local_record["station_id"],
                    local_record["timestamp"],
                    local_record["anomaly_type"],
                    local_record["final_status"],
                    local_record["severity"],
                    local_record["affected_parameter"],
                    local_record["observed_value"],
                    local_record["expected_value"],
                    local_record["anomaly_score"],
                    local_record["confidence"],
                    local_record["evidence"],
                    local_record["explanation"],
                    local_record["model_name"],
                    local_record["model_version"],
                    local_record["status"],
                    local_record["first_detected"],
                    local_record["last_detected"],
                    local_record["occurrence_count"],
                    local_record["session_id"],
                    local_record["created_at"],
                    local_record["resolved_at"],
                ),
            )
            local_record["id"] = cursor.lastrowid
            conn.commit()
    except Exception as sq_exc:
        logger.warning(f"[DatabaseService] SQLite anomaly insert notice: {sq_exc}")

    # 2. Supabase persistence
    try:
        payload = {
            "station_id": local_record["station_id"],
            "timestamp": local_record["timestamp"],
            "anomaly_type": local_record["anomaly_type"],
            "final_status": local_record["final_status"],
            "severity": local_record["severity"],
            "affected_parameter": local_record["affected_parameter"],
            "observed_value": local_record["observed_value"],
            "expected_value": local_record["expected_value"],
            "anomaly_score": local_record["anomaly_score"],
            "confidence": local_record["confidence"],
            "evidence": evidence_dict,
            "explanation": local_record["explanation"],
            "model_name": local_record["model_name"],
            "model_version": local_record["model_version"],
            "status": local_record["status"],
            "first_detected": local_record["first_detected"],
            "last_detected": local_record["last_detected"],
            "occurrence_count": local_record["occurrence_count"],
            "session_id": local_record["session_id"],
            "resolved_at": local_record["resolved_at"],
        }
        remote_cols = _get_remote_columns("anomalies")
        if remote_cols:
            payload = {k: v for k, v in payload.items() if k in remote_cols}

        res = supabase.table("anomalies").insert(payload).execute()
        if res.data and len(res.data) > 0:
            merged = dict(local_record)
            merged.update(res.data[0])
            return merged
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase anomaly insert notice: {exc}")

    return local_record


def update_anomaly_status(
    anomaly_id: Any,
    status: str,
    resolved_at: Optional[Any] = None,
) -> Optional[dict[str, Any]]:
    """Update lifecycle status of an anomaly (e.g. DETECTED -> ACKNOWLEDGED -> RESOLVED)."""
    status_upper = str(status).strip().upper()
    now_iso = datetime.now(timezone.utc).isoformat()
    resolved_iso = _format_timestamp(resolved_at or now_iso) if status_upper == "RESOLVED" else None

    # 1. Update SQLite
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE anomalies_local SET status = ?, resolved_at = ? WHERE id = ? OR reading_id = ?",
                (status_upper, resolved_iso, str(anomaly_id), str(anomaly_id)),
            )
            conn.commit()
    except Exception as sq_exc:
        logger.warning(f"[DatabaseService] SQLite anomaly update notice: {sq_exc}")

    # 2. Update Supabase
    try:
        update_data: dict[str, Any] = {"status": status_upper}
        if resolved_iso:
            update_data["resolved_at"] = resolved_iso

        remote_cols = _get_remote_columns("anomalies")
        if remote_cols:
            update_data = {k: v for k, v in update_data.items() if k in remote_cols}

        res = supabase.table("anomalies").update(update_data).eq("id", anomaly_id).execute()
        if res.data and len(res.data) > 0:
            return res.data[0]
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase anomaly status update notice: {exc}")

    return {"id": anomaly_id, "status": status_upper, "resolved_at": resolved_iso}


def get_recent_anomalies(
    limit: int = 20,
    station_id: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    include_demo: bool = False,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve recent detected anomalies."""
    # 1. Try Supabase
    try:
        query = supabase.table("anomalies").select("*").order("timestamp", desc=True).limit(limit)
        if station_id:
            query = query.eq("station_id", station_id)
        if status:
            query = query.eq("status", status.upper())
        if severity:
            query = query.eq("severity", severity.upper())
        if session_id:
            query = query.eq("session_id", session_id)

        res = query.execute()
        if res.data and len(res.data) > 0:
            return res.data
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase anomaly query notice: {exc}")

    # 2. Fallback to SQLite
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            clauses = []
            params: list[Any] = []
            if station_id:
                clauses.append("station_id = ?")
                params.append(station_id)
            if status:
                clauses.append("status = ?")
                params.append(status.upper())
            if severity:
                clauses.append("severity = ?")
                params.append(severity.upper())
            if session_id:
                clauses.append("session_id = ?")
                params.append(session_id)

            where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            sql = f"SELECT * FROM anomalies_local {where_sql} ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, tuple(params))
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                if d.get("evidence") and isinstance(d["evidence"], str):
                    try:
                        d["evidence"] = json.loads(d["evidence"])
                    except Exception:
                        pass
                rows.append(d)
            return rows
    except Exception as sq_exc:
        logger.error(f"[DatabaseService] SQLite anomaly fetch error: {sq_exc}")
        return []


# =====================================================================
# 3. STATION HEALTH AUDIT PERSISTENCE
# =====================================================================

def save_station_health(
    station_id: str,
    health_score: int,
    status: str,
    issues: Optional[list[str]] = None,
    timestamp: Any = None,
    availability: float = 100.0,
    expected_observations: int = 0,
    received_observations: int = 0,
    missing_observations: int = 0,
    anomaly_frequency: float = 0.0,
    false_alarm_metrics: Optional[dict[str, Any]] = None,
    last_seen: Any = None,
) -> Optional[dict[str, Any]]:
    """Persist station health score, availability, and observation telemetry counters."""
    ts_iso = _format_timestamp(timestamp)
    last_seen_iso = _format_timestamp(last_seen or timestamp)
    now_iso = datetime.now(timezone.utc).isoformat()
    issues_list = issues or []
    issues_json = json.dumps(issues_list)
    fa_metrics = false_alarm_metrics or {}
    fa_json = json.dumps(fa_metrics)

    local_record: dict[str, Any] = {
        "station_id": str(station_id).strip(),
        "timestamp": ts_iso,
        "availability": float(availability),
        "expected_observations": int(expected_observations),
        "received_observations": int(received_observations),
        "missing_observations": int(missing_observations),
        "anomaly_frequency": float(anomaly_frequency),
        "false_alarm_metrics": fa_json,
        "health_score": int(health_score),
        "status": str(status).strip().upper(),
        "issues": issues_json,
        "last_seen": last_seen_iso,
        "created_at": now_iso,
        "updated_at": now_iso,
    }

    # 1. SQLite persistence
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO station_health_local (
                    station_id, timestamp, availability, expected_observations,
                    received_observations, missing_observations, anomaly_frequency,
                    false_alarm_metrics, health_score, status, issues, last_seen,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    local_record["station_id"],
                    local_record["timestamp"],
                    local_record["availability"],
                    local_record["expected_observations"],
                    local_record["received_observations"],
                    local_record["missing_observations"],
                    local_record["anomaly_frequency"],
                    local_record["false_alarm_metrics"],
                    local_record["health_score"],
                    local_record["status"],
                    local_record["issues"],
                    local_record["last_seen"],
                    local_record["created_at"],
                    local_record["updated_at"],
                ),
            )
            local_record["id"] = cursor.lastrowid
            conn.commit()
    except Exception as sq_exc:
        logger.warning(f"[DatabaseService] SQLite station health notice: {sq_exc}")

    # 2. Supabase persistence
    try:
        payload = {
            "station_id": local_record["station_id"],
            "timestamp": local_record["timestamp"],
            "health_score": local_record["health_score"],
            "status": local_record["status"],
            "availability": local_record["availability"],
            "expected_observations": local_record["expected_observations"],
            "received_observations": local_record["received_observations"],
            "missing_observations": local_record["missing_observations"],
            "anomaly_frequency": local_record["anomaly_frequency"],
            "false_alarm_metrics": fa_metrics,
            "issues": issues_list,
            "last_seen": local_record["last_seen"],
        }
        remote_cols = _get_remote_columns("station_health")
        if remote_cols:
            payload = {k: v for k, v in payload.items() if k in remote_cols}

        res = supabase.table("station_health").insert(payload).execute()
        if res.data and len(res.data) > 0:
            merged = dict(local_record)
            merged.update(res.data[0])
            return merged
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase station health notice: {exc}")

    return local_record


def get_station_health(station_id: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve latest station health records."""
    # 1. Try Supabase
    try:
        query = supabase.table("station_health").select("*").order("timestamp", desc=True)
        if station_id:
            query = query.eq("station_id", station_id).limit(1)
        else:
            query = query.limit(limit)
        res = query.execute()
        if res.data and len(res.data) > 0:
            return res.data
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase station health query notice: {exc}")

    # 2. Fallback to SQLite
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if station_id:
                cursor.execute(
                    "SELECT * FROM station_health_local WHERE station_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (station_id,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM station_health_local ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            rows = []
            for r in cursor.fetchall():
                d = dict(r)
                if d.get("issues") and isinstance(d["issues"], str):
                    try:
                        d["issues"] = json.loads(d["issues"])
                    except Exception:
                        pass
                if d.get("false_alarm_metrics") and isinstance(d["false_alarm_metrics"], str):
                    try:
                        d["false_alarm_metrics"] = json.loads(d["false_alarm_metrics"])
                    except Exception:
                        pass
                rows.append(d)
            return rows
    except Exception as sq_exc:
        logger.error(f"[DatabaseService] SQLite health fetch error: {sq_exc}")
        return []


# =====================================================================
# 4. MODEL VERSIONS PERSISTENCE & GOVERNANCE
# =====================================================================

def save_model_version(model_data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Register or update an ML model version in model_versions table."""
    model_name = str(model_data.get("model_name", "SkyGuard-LOF-BME280")).strip()
    model_version = str(model_data.get("model_version", "v1.0")).strip()
    algorithm = str(model_data.get("algorithm", "LocalOutlierFactor")).strip()
    dataset_version = str(model_data.get("dataset_version", "benchmark_v1.0_clean")).strip()
    feature_schema_version = str(model_data.get("feature_schema_version", "features_v1.0")).strip()
    threshold = float(model_data.get("threshold", 1.50))
    training_meta = model_data.get("training_metadata") or {}
    eval_metrics = model_data.get("evaluation_metrics") or {}
    artifact_hash = model_data.get("artifact_hash")
    now_iso = datetime.now(timezone.utc).isoformat()

    local_record = {
        "model_name": model_name,
        "model_version": model_version,
        "algorithm": algorithm,
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "threshold": threshold,
        "training_metadata": json.dumps(training_meta),
        "evaluation_metrics": json.dumps(eval_metrics),
        "artifact_hash": artifact_hash,
        "created_at": now_iso,
    }

    # 1. SQLite persistence
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO model_versions_local (
                    model_name, model_version, algorithm, dataset_version,
                    feature_schema_version, threshold, training_metadata,
                    evaluation_metrics, artifact_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_version) DO UPDATE SET
                    model_name=excluded.model_name,
                    algorithm=excluded.algorithm,
                    dataset_version=excluded.dataset_version,
                    feature_schema_version=excluded.feature_schema_version,
                    threshold=excluded.threshold,
                    training_metadata=excluded.training_metadata,
                    evaluation_metrics=excluded.evaluation_metrics,
                    artifact_hash=excluded.artifact_hash
                """,
                (
                    local_record["model_name"],
                    local_record["model_version"],
                    local_record["algorithm"],
                    local_record["dataset_version"],
                    local_record["feature_schema_version"],
                    local_record["threshold"],
                    local_record["training_metadata"],
                    local_record["evaluation_metrics"],
                    local_record["artifact_hash"],
                    local_record["created_at"],
                ),
            )
            local_record["id"] = cursor.lastrowid
            conn.commit()
    except Exception as sq_exc:
        logger.warning(f"[DatabaseService] SQLite model_versions notice: {sq_exc}")

    # 2. Supabase persistence
    try:
        remote_cols = _get_remote_columns("model_versions")
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "algorithm": algorithm,
            "dataset_version": dataset_version,
            "feature_schema_version": feature_schema_version,
            "threshold": threshold,
            "training_metadata": training_meta,
            "evaluation_metrics": eval_metrics,
            "artifact_hash": artifact_hash,
        }
        if "version" in remote_cols and "model_version" not in remote_cols:
            payload["version"] = model_version
        if "metrics" in remote_cols and "evaluation_metrics" not in remote_cols:
            payload["metrics"] = eval_metrics
        if "artifact_path" in remote_cols and "artifact_hash" not in remote_cols:
            payload["artifact_path"] = artifact_hash or "skyguard_lof_bme280_calibrated.joblib"

        if remote_cols:
            payload = {k: v for k, v in payload.items() if k in remote_cols}

        res = supabase.table("model_versions").upsert(payload).execute()
        if res.data and len(res.data) > 0:
            ret = dict(res.data[0])
            if "version" in ret and "model_version" not in ret:
                ret["model_version"] = ret["version"]
            if "metrics" in ret and "evaluation_metrics" not in ret:
                ret["evaluation_metrics"] = ret["metrics"]
            return ret
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase model_versions notice: {exc}")

    return local_record


def get_model_versions(limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve all registered ML models with evaluation metrics."""
    # Read local SQLite map to enrich fields if cloud migrations are pending
    sqlite_map: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM model_versions_local")
            for r in c.fetchall():
                d = dict(r)
                for f in ("training_metadata", "evaluation_metrics"):
                    if d.get(f) and isinstance(d[f], str):
                        try:
                            d[f] = json.loads(d[f])
                        except Exception:
                            pass
                sqlite_map[d["model_version"]] = d
    except Exception:
        pass

    # 1. Try Supabase
    try:
        res = supabase.table("model_versions").select("*").order("created_at", desc=True).limit(limit).execute()
        if res.data and len(res.data) > 0:
            standardized = []
            for m in res.data:
                item = dict(m)
                if "version" in item and "model_version" not in item:
                    item["model_version"] = item["version"]
                if "metrics" in item and "evaluation_metrics" not in item:
                    item["evaluation_metrics"] = item["metrics"]

                mv_key = item.get("model_version")
                if mv_key and mv_key in sqlite_map:
                    enriched = dict(sqlite_map[mv_key])
                    enriched.update({k: v for k, v in item.items() if v is not None})
                    standardized.append(enriched)
                else:
                    item.setdefault("algorithm", "LocalOutlierFactor")
                    item.setdefault("dataset_version", "benchmark_v1.0_clean")
                    item.setdefault("threshold", 1.50)
                    standardized.append(item)
            return standardized
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase model_versions fetch notice: {exc}")

    # 2. Fallback to SQLite
    return list(sqlite_map.values())[:limit]


# =====================================================================
# 5. STATIONS INVENTORY PERSISTENCE & SEEDING
# =====================================================================

def seed_stations_inventory(stations_list: list[dict[str, Any]]) -> int:
    """Seed or update the stations inventory table in both Supabase and SQLite."""
    seeded_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for s in stations_list:
        stn_id = str(s.get("station_id") or s.get("id")).strip()
        name = str(s.get("name") or stn_id).strip()
        lat = float(s.get("latitude") or s.get("lat") or 0.0)
        lon = float(s.get("longitude") or s.get("lon") or 0.0)
        state = s.get("state")
        district = s.get("district")
        source_type = s.get("source_type") or s.get("station_type") or "IMD_AWS_REFERENCE"
        status = str(s.get("status") or "ACTIVE").upper()
        is_sim = 1 if s.get("is_simulated") else 0

        # 1. Persist to SQLite
        try:
            with sqlite3.connect(SQLITE_DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO stations_local (
                        station_id, name, latitude, longitude, state, district,
                        source_type, status, is_simulated, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(station_id) DO UPDATE SET
                        name=excluded.name,
                        latitude=excluded.latitude,
                        longitude=excluded.longitude,
                        state=excluded.state,
                        district=excluded.district,
                        source_type=excluded.source_type,
                        status=excluded.status,
                        is_simulated=excluded.is_simulated,
                        updated_at=excluded.updated_at
                    """,
                    (stn_id, name, lat, lon, state, district, source_type, status, is_sim, now_iso, now_iso),
                )
                conn.commit()
                seeded_count += 1
        except Exception as sq_exc:
            logger.debug(f"[DatabaseService] SQLite station seed notice: {sq_exc}")

        # 2. Persist to Supabase
        try:
            payload = {
                "station_id": stn_id,
                "name": name,
                "latitude": lat,
                "longitude": lon,
                "state": state,
                "district": district,
                "source_type": source_type,
                "status": status,
                "is_simulated": bool(is_sim),
            }
            remote_cols = _get_remote_columns("stations")
            if remote_cols:
                payload = {k: v for k, v in payload.items() if k in remote_cols}

            supabase.table("stations").upsert(payload, on_conflict="station_id").execute()
        except Exception as exc:
            logger.debug(f"[DatabaseService] Supabase station seed notice: {exc}")

    return seeded_count


def get_stations(status: Optional[str] = None) -> list[dict[str, Any]]:
    """Retrieve stations list from Supabase with SQLite fallback."""
    # Read local SQLite map
    sqlite_map: dict[str, dict[str, Any]] = {}
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if status:
                cursor.execute("SELECT * FROM stations_local WHERE status = ? ORDER BY station_id", (status.upper(),))
            else:
                cursor.execute("SELECT * FROM stations_local ORDER BY station_id")
            for r in cursor.fetchall():
                d = dict(r)
                d["is_simulated"] = bool(d.get("is_simulated", 0))
                sqlite_map[d["station_id"]] = d
    except Exception:
        pass

    # 1. Try Supabase
    try:
        query = supabase.table("stations").select("*").order("station_id")
        if status:
            query = query.eq("status", status.upper())
        res = query.execute()
        if res.data and len(res.data) > 0:
            stations_out = []
            for s in res.data:
                item = dict(s)
                sid = item.get("station_id")
                if sid and sid in sqlite_map:
                    enriched = dict(sqlite_map[sid])
                    enriched.update({k: v for k, v in item.items() if v is not None})
                    stations_out.append(enriched)
                else:
                    if "is_simulated" not in item:
                        item["is_simulated"] = not (sid in ("43189", "43150", "43245"))
                    stations_out.append(item)
            return stations_out
    except Exception as exc:
        logger.debug(f"[DatabaseService] Supabase stations fetch notice: {exc}")

    # 2. Fallback to SQLite
    return list(sqlite_map.values())
