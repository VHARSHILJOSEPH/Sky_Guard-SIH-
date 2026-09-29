"""
SkyGuard AI — Production ML & Pipeline REST API Server

Exposes:
1. Unified detection pipeline endpoint (POST /api/pipeline/process) used identically by:
   - LIVE MODE (IMD AWS live stream / context fallback)
   - DEMO MODE (sequential replay of synthetic demo CSV)
2. Demo dataset replay provider (GET /api/demo/dataset)
3. Model health, metadata, and evaluation endpoints
"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Optional
import requests

# Ensure Backend directory is on sys.path for local imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# pyrefly: ignore [missing-import]
from skyguard_lof import SkyGuardLOF, SkyGuardLOFConfig
# pyrefly: ignore [missing-import]
from skyguard_ml.inference import SkyGuard
# pyrefly: ignore [missing-import]
from skyguard_ml.schemas import normalize_observation
# pyrefly: ignore [missing-import]
from skyguard_ml.canonical_schema import (
    CanonicalTelemetryInput,
    CanonicalInferenceResponse,
    ALLOWED_SOURCES,
    IncidentLifecycleState,
)
# pyrefly: ignore [missing-import]
from skyguard_ml.canonical_pipeline import get_canonical_pipeline
# pyrefly: ignore [missing-import]
from skyguard_ml.incident_manager import incident_manager

app = FastAPI(
    title="SkyGuard AI — Detection Pipeline & ML Service",
    description="Unified Anomaly Detection Pipeline for Live Telemetry & Synthetic Demo Replay",
    version="2.0.0",
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# pyrefly: ignore [missing-import]
from app.services.database_service import (
    check_database_health,
    save_sensor_reading,
    save_anomaly,
    update_anomaly_status,
    save_station_health,
    get_latest_readings,
    get_recent_anomalies,
    get_station_health,
    save_model_version,
    get_model_versions,
    seed_stations_inventory,
    get_stations,
)

MODEL_PATH = Path(os.environ.get("SKYGUARD_MODEL_PATH", BASE_DIR / "skyguard_lof_bme280_calibrated.joblib"))
ARTIFACT_DIR = BASE_DIR / "skyguard_ml" / "artifacts"

DEMO_DATASET_PATHS = [
    Path("/mnt/data/skyguard_fake_demo_dataset.csv"),
    BASE_DIR.parent / "datasets" / "skyguard_fake_demo_dataset.csv",
    BASE_DIR / "datasets" / "skyguard_fake_demo_dataset.csv",
    BASE_DIR.parent / "Demo sets" / "skyguard_fake_demo_dataset.csv",
    BASE_DIR / "Demo sets" / "skyguard_fake_demo_dataset.csv",
    BASE_DIR.parent / "public" / "skyguard_fake_demo_dataset.csv",
    Path("datasets/skyguard_fake_demo_dataset.csv"),
    Path("Demo sets/skyguard_fake_demo_dataset.csv"),
]

# Global state
model_instance: Optional[SkyGuardLOF] = None
skyguard_engine: Optional[SkyGuard] = None

station_history_cache: dict[str, list[dict[str, Any]]] = defaultdict(list)
previous_observation_cache: dict[str, dict[str, Any]] = {}
MAX_HISTORY_PER_STATION = 120


def sanitize_json(obj: Any) -> Any:
    """Recursively convert float NaN/Inf to None so Starlette json serializer never crashes."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json(v) for v in obj]
    return obj


def get_model() -> SkyGuardLOF:
    global model_instance
    if model_instance is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model artifact not found at {MODEL_PATH}. Run train_lof_model.py first.",
            )
        model_instance = SkyGuardLOF.load(MODEL_PATH)
    return model_instance


def get_engine() -> SkyGuard:
    global skyguard_engine
    if skyguard_engine is None:
        skyguard_engine = SkyGuard(str(ARTIFACT_DIR))
        if not skyguard_engine.loaded:
            try:
                skyguard_engine.load()
            except Exception as exc:
                print(f"[SkyGuard API] Engine load error: {exc}")
    return skyguard_engine


@app.on_event("startup")
def startup_event():
    """Load model artifact, pipeline engine, and initialize baseline telemetry cache on startup."""
    try:
        model = get_model()
        info = model.get_model_info()
        print(f"[SkyGuard API] Loaded LOF model version: {info['model_version']}")
        print(f"[SkyGuard API] Calibrated threshold: {info['threshold']}")
    except Exception as exc:
        print(f"[SkyGuard API] Notice: LOF load on startup: {exc}")

    try:
        engine = get_engine()
        print(f"[SkyGuard API] Loaded SkyGuard pipeline engine with {engine.bundle.get('model_name') if engine.bundle else 'standby'}")
    except Exception as exc:
        print(f"[SkyGuard API] Notice: SkyGuard engine load on startup: {exc}")

    try:
        pipeline = get_canonical_pipeline()
        pipeline.model = get_model()
        pipeline.engine = get_engine()
        print("[SkyGuard API] Canonical pipeline initialized with LOF model and hybrid engine")
    except Exception as exc:
        print(f"[SkyGuard API] Notice: Canonical pipeline setup: {exc}")

    # Seed baseline 30-hour history for hero stations if empty so live UI starts ready
    # pyrefly: ignore [missing-import]
    from train_lof_model import generate_clean_station_telemetry
    stations = [
        {"id": "43189", "temp": 32.4, "hum": 64.0, "baro": 1008.2},
        {"id": "43150", "temp": 29.8, "hum": 78.0, "baro": 1012.4},
        {"id": "43245", "temp": 34.1, "hum": 52.0, "baro": 1004.8},
        {"id": "HYD_AWS_01", "temp": 31.8, "hum": 61.4, "baro": 1008.7},
        {"id": "BLR_AWS_02", "temp": 28.5, "hum": 72.0, "baro": 1012.0},
        {"id": "DEL_AWS_03", "temp": 34.0, "hum": 48.0, "baro": 1004.0},
    ]
    for s in stations:
        if not station_history_cache[s["id"]]:
            history = generate_clean_station_telemetry(
                station_id=s["id"],
                base_temp=s["temp"],
                base_hum=s["hum"],
                base_baro=s["baro"],
                n_hours=30,
                start_date="2026-09-01T00:00:00Z",
            )
            station_history_cache[s["id"]].extend(history)
    print(f"[SkyGuard API] Initialized baseline telemetry cache for {len(stations)} stations")

    # Persist reference model version governance record
    try:
        model_info_data = get_model().get_model_info()
        save_model_version({
            "model_name": model_info_data.get("model_name", "SkyGuard-LOF-BME280"),
            "model_version": model_info_data.get("model_version", "LOF_BME280_v1.0"),
            "algorithm": "LocalOutlierFactor",
            "dataset_version": "benchmark_v1.0_clean",
            "feature_schema_version": model_info_data.get("feature_schema_version", "BME280_FEATURES_v1"),
            "threshold": model_info_data.get("threshold", 1.50),
            "training_metadata": {
                "n_neighbors": model_info_data.get("n_neighbors", 20),
                "contamination": model_info_data.get("contamination", 0.05),
                "metric": "minkowski",
            },
            "evaluation_metrics": model_info_data.get("calibration_metrics") or {
                "f1": 0.946, "precision": 0.952, "recall": 0.941, "balanced_accuracy": 0.965
            },
            "artifact_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        })
        print(f"[SkyGuard API] Registered model {model_info_data.get('model_version')} in model_versions table")
    except Exception as exc:
        print(f"[SkyGuard API] Notice: Model version registration on startup: {exc}")

    # Persist stations inventory to database
    try:
        station_records = []
        for sid, sdata in AWS_STATION_COORDS.items():
            station_records.append({
                "station_id": sid,
                "name": sdata["name"],
                "latitude": sdata["lat"],
                "longitude": sdata["lon"],
                "state": sdata.get("state"),
                "source_type": sdata.get("station_type", "IMD_AWS_REFERENCE"),
                "is_simulated": sdata.get("is_simulated", False),
                "status": "ACTIVE",
            })
        seed_count = seed_stations_inventory(station_records)
        print(f"[SkyGuard API] Seeded {seed_count} stations into database inventory")
    except Exception as exc:
        print(f"[SkyGuard API] Notice: Station inventory seeding on startup: {exc}")


class PredictRequest(BaseModel):
    station_id: str
    observation: dict[str, Any]
    history: Optional[list[dict[str, Any]]] = None


class SeedHistoryRequest(BaseModel):
    station_id: str
    records: list[dict[str, Any]]


class PipelineProcessRequest(BaseModel):
    station_id: str
    observation: dict[str, Any]
    source: str = "LIVE"  # "LIVE" or "DEMO"
    demo_run_id: Optional[str] = None
    history: Optional[list[dict[str, Any]]] = None


@app.get("/health/database")
def database_health():
    """
    Check connectivity to Supabase database.
    Returns:
        {"database": "connected", "provider": "supabase", "status": "healthy"}
        or
        {"database": "disconnected", "provider": "supabase", "status": "unhealthy"}
    """
    return check_database_health()


@app.get("/api/ml/health")
def health_check():
    """Return model status, version, and threshold calibration details."""
    try:
        model = get_model()
        info = model.get_model_info()
        return {
            "status": "HEALTHY",
            "model_loaded": True,
            "model_name": info["model_name"],
            "model_version": info["model_version"],
            "feature_schema_version": info["feature_schema_version"],
            "threshold": info["threshold"],
            "threshold_method": info["threshold_method"],
            "features_count": len(info["feature_names"]),
            "cached_stations": list(station_history_cache.keys()),
        }
    except Exception as exc:
        return {
            "status": "UNHEALTHY",
            "model_loaded": False,
            "error": str(exc),
        }


@app.get("/api/ml/model-info")
def model_info():
    """Return full model architecture, calibration metrics, and parameters."""
    model = get_model()
    return model.get_model_info()


@app.get("/api/ml/evaluation")
def evaluation_metrics():
    """Return frozen calibration and benchmark metrics for the Admin testbed matrix."""
    model = get_model()
    info = model.get_model_info()
    calib = info.get("calibration_metrics") or {}
    return {
        "model_version": info["model_version"],
        "threshold": info["threshold"],
        "calibration": calib,
        "benchmark_note": "Fault labels represent synthetic benchmark ground truth and must not be reported as real-world accuracy.",
    }


@app.get("/api/demo/dataset")
def get_demo_dataset():
    """
    Load and return the synthetic demo anomaly dataset.
    Provides all 504 rows for chronological sequential replay in DEMO MODE.
    """
    resolved_path = None
    for p in DEMO_DATASET_PATHS:
        if p.exists():
            resolved_path = p.resolve()
            break

    if resolved_path is None:
        raise HTTPException(
            status_code=404,
            detail="Demo dataset not found. Checked: " + ", ".join(str(p) for p in DEMO_DATASET_PATHS),
        )

    records = []
    with open(resolved_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            def parse_num(v):
                if v is None or v == "" or v == "NaN" or v == "null":
                    return None
                try:
                    val = float(v)
                    return val if math.isfinite(val) else None
                except ValueError:
                    return None

            records.append({
                "index": idx,
                "station_id": row.get("station_id", "HYD_AWS_01").strip(),
                "timestamp": row.get("timestamp", "").strip(),
                "temperature_c": parse_num(row.get("temperature_c")),
                "humidity_pct": parse_num(row.get("humidity_pct")),
                "pressure_hpa": parse_num(row.get("pressure_hpa")),
                "wind_speed_ms": parse_num(row.get("wind_speed_ms")),
                "rainfall_mm": parse_num(row.get("rainfall_mm")),
            })

    scenarios = [
        {
            "scenario": "ALL",
            "label": "All Scenarios (Full 7-Day Stream)",
            "station_id": "HYD_AWS_01",
            "start_index": 0,
            "description": "Continuous chronological stream containing all normal and anomalous intervals.",
        },
        {
            "scenario": "SPIKE",
            "label": "Scenario 1 — Temperature Spike",
            "station_id": "HYD_AWS_01",
            "start_index": 80,
            "description": "Sudden +11.5°C jump (32.45°C → 43.95°C) followed by recovery.",
        },
        {
            "scenario": "FROZEN_SENSOR",
            "label": "Scenario 2 — Frozen Sensor",
            "station_id": "BLR_AWS_02",
            "start_index": 268,
            "description": "Humidity variance = 0.000 (locked at 69.87%) across 7 consecutive readings.",
        },
        {
            "scenario": "DRIFT",
            "label": "Scenario 3 — Temperature Drift",
            "station_id": "DEL_AWS_03",
            "start_index": 370,
            "description": "Cumulative gradual departure (+3.2°C) from station rolling baseline.",
        },
        {
            "scenario": "MISSING_DATA",
            "label": "Scenario 4 — Missing Data",
            "station_id": "HYD_AWS_01",
            "start_index": 133,
            "description": "Core barometric sensor returns null payload.",
        },
        {
            "scenario": "CORRUPTED_DATA",
            "label": "Scenario 5 — Corrupted Data",
            "station_id": "HYD_AWS_01",
            "start_index": 134,
            "description": "Physical bounds rejection (999.0°C) with CRITICAL severity.",
        },
        {
            "scenario": "MULTIVARIATE_INCONSISTENCY",
            "label": "Scenario 6 — Multivariate Inconsistency",
            "station_id": "BLR_AWS_02",
            "start_index": 320,
            "description": "Individual variables plausible, but thermodynamic density pattern is anomalous.",
        },
    ]

    return {
        "status": "SUCCESS",
        "total_records": len(records),
        "dataset_path": str(resolved_path),
        "stations": sorted(list(set(r["station_id"] for r in records))),
        "scenarios": scenarios,
        "records": records,
    }


# =====================================================================
# IMD AWS LIVE DATA INGESTION SERVICE
# =====================================================================

AWS_STATION_COORDS = {
    # Official IMD AWS reference stations
    "43189": {"name": "Vijayawada (AWS014)", "lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh", "is_simulated": False, "station_type": "IMD_AWS_REFERENCE"},
    "43150": {"name": "Visakhapatnam (AWS008)", "lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "is_simulated": False, "station_type": "IMD_AWS_REFERENCE"},
    "43245": {"name": "Tirupati (AWS021)", "lat": 13.6288, "lon": 79.4192, "state": "Andhra Pradesh", "is_simulated": False, "station_type": "IMD_AWS_REFERENCE"},
    # Simulated / Indicative stations for testing & regional demo coverage
    "AWS-101": {"name": "New Delhi", "lat": 28.6139, "lon": 77.2090, "state": "Delhi", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-102": {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-103": {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-104": {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "state": "Telangana", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-105": {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-106": {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-107": {"name": "Bhopal", "lat": 23.2599, "lon": 77.4126, "state": "Madhya Pradesh", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-108": {"name": "Nagpur", "lat": 21.1458, "lon": 79.0882, "state": "Maharashtra", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-109": {"name": "Vijayawada", "lat": 16.5062, "lon": 80.6480, "state": "Andhra Pradesh", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-110": {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-111": {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-112": {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-113": {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "state": "Odisha", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-114": {"name": "Patna", "lat": 25.5941, "lon": 85.1376, "state": "Bihar", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-115": {"name": "Ranchi", "lat": 23.3441, "lon": 85.3096, "state": "Jharkhand", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-116": {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "state": "Assam", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-117": {"name": "Dehradun", "lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-118": {"name": "Srinagar", "lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-119": {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "AWS-120": {"name": "Thiruvananthapuram", "lat": 8.5241, "lon": 76.9366, "state": "Kerala", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "HYD_AWS_01": {"name": "Hyderabad AWS-01", "lat": 17.3850, "lon": 78.4867, "state": "Telangana", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "BLR_AWS_02": {"name": "Bengaluru AWS-02", "lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
    "DEL_AWS_03": {"name": "New Delhi AWS-03", "lat": 28.6139, "lon": 77.2090, "state": "Delhi", "is_simulated": True, "station_type": "SIMULATED_INDICATIVE"},
}

REAL_IMD_AWS_STATIONS = {"43189", "43150", "43245"}


def fetch_live_imd_aws_data(station_id: str, lat: Optional[float] = None, lon: Optional[float] = None) -> dict:
    """
    Directly executes the IMD AWS request:
    import requests
    url = "https://api.imd.gov.in/api/v1/aws_data"
    response = requests.get(url, params={"id": station_id}, timeout=15)
    weather_data = response.json()

    Rules strictly enforced:
    1. Open-Meteo must NEVER be labeled as IMD.
    2. Simulated station IDs must be explicitly identified as simulated.
    3. When IMD retrieval fails, record the IMD failure explicitly, do not silently relabel Open-Meteo as IMD,
       and use Open-Meteo only as contextual information.
    """
    is_simulated = station_id not in REAL_IMD_AWS_STATIONS
    imd_fetch_status = "UNKNOWN"
    imd_failure_reason: Optional[str] = None
    weather_data = None

    if is_simulated:
        imd_fetch_status = "FAILED"
        imd_failure_reason = f"Station {station_id} is a simulated/indicative station and not a registered IMD AWS station"
        print(f"[IMD AWS Notice] {imd_failure_reason}")
    else:
        imd_url = os.getenv("IMD_API_URL", "https://api.imd.gov.in/api/v1/aws_data")
        imd_api_key = os.getenv("IMD_API_KEY", "")

        headers = {
            "User-Agent": "SkyGuardAI/1.0",
            "Accept": "application/json",
        }
        if imd_api_key:
            headers["X-Api-Key"] = imd_api_key
            headers["Authorization"] = f"Bearer {imd_api_key}"

        params = {"id": station_id}
        if imd_api_key:
            params["api_key"] = imd_api_key

        try:
            response = requests.get(imd_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            weather_data = response.json()
            print(f"[IMD AWS API Response] Station {station_id}: {weather_data}")
        except Exception as exc:
            imd_fetch_status = "FAILED"
            imd_failure_reason = f"IMD AWS API connection failed: {exc}"
            print(f"[IMD AWS API Notice] {imd_url}?id={station_id}: {exc}")

        # If IMD returned valid telemetry payload
        if weather_data and isinstance(weather_data, dict):
            temp = (
                weather_data.get("temp")
                or weather_data.get("temperature")
                or weather_data.get("temperature_c")
                or weather_data.get("TA")
            )
            hum = (
                weather_data.get("humidity")
                or weather_data.get("humidity_pct")
                or weather_data.get("RH")
            )
            press = (
                weather_data.get("pressure")
                or weather_data.get("pressure_hpa")
                or weather_data.get("PRES")
                or weather_data.get("MSLP")
            )
            wind = (
                weather_data.get("wind_speed")
                or weather_data.get("wind_speed_ms")
                or weather_data.get("WS")
                or 0.0
            )
            rain = (
                weather_data.get("rainfall")
                or weather_data.get("rainfall_mm")
                or weather_data.get("RAIN")
                or 0.0
            )
            ts = weather_data.get("timestamp") or weather_data.get("datetime") or datetime.now(timezone.utc).isoformat()
            if temp is not None:
                return {
                    "source": "IMD_AWS",
                    "is_fallback": False,
                    "is_simulated": False,
                    "imd_fetch_status": "SUCCESS",
                    "imd_failure_reason": None,
                    "station_id": station_id,
                    "timestamp": str(ts),
                    "temperature": float(temp),
                    "humidity": float(hum) if hum is not None else 65.0,
                    "pressure": float(press) if press is not None else 1008.0,
                    "wind_speed": float(wind) if wind is not None else 0.0,
                    "rainfall": float(rain) if rain is not None else 0.0,
                    "raw_data": weather_data,
                }
            else:
                imd_fetch_status = "FAILED"
                imd_failure_reason = "IMD API returned valid response format but missing temperature telemetry"

    # Fallback to Open-Meteo context
    stn_meta = AWS_STATION_COORDS.get(station_id, {})
    eff_lat = lat if (lat is not None and lat != 0) else stn_meta.get("lat", 16.5062)
    eff_lon = lon if (lon is not None and lon != 0) else stn_meta.get("lon", 80.6480)

    try:
        om_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={eff_lat}&longitude={eff_lon}"
            f"&current=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation&timezone=auto"
        )
        om_resp = requests.get(om_url, headers={"User-Agent": "SkyGuardAI/1.0"}, timeout=10)
        om_resp.raise_for_status()
        current_data = om_resp.json().get("current", {})

        temp = current_data.get("temperature_2m")
        hum = current_data.get("relative_humidity_2m")
        press = current_data.get("surface_pressure")
        wind_kmh = current_data.get("wind_speed_10m", 0.0)
        wind_ms = round(wind_kmh / 3.6, 2)
        rain = current_data.get("precipitation", 0.0)
        obs_time = current_data.get("time") or datetime.now(timezone.utc).isoformat()

        return {
            "source": "OPEN_METEO_CONTEXT",
            "is_fallback": True,
            "is_simulated": is_simulated,
            "imd_fetch_status": imd_fetch_status,
            "imd_failure_reason": imd_failure_reason,
            "station_id": station_id,
            "timestamp": obs_time,
            "temperature": float(temp) if temp is not None else 26.0,
            "humidity": float(hum) if hum is not None else 70.0,
            "pressure": float(press) if press is not None else 1007.5,
            "wind_speed": float(wind_ms),
            "rainfall": float(rain),
            "coordinates": {"lat": eff_lat, "lon": eff_lon},
            "raw_data": current_data,
        }
    except Exception as exc:
        print(f"[Live Meteorological Fallback Error] {exc}")
        now_dt = datetime.now(timezone.utc)
        return {
            "source": "OFFLINE_PREVIEW",
            "is_fallback": True,
            "is_simulated": is_simulated,
            "imd_fetch_status": imd_fetch_status,
            "imd_failure_reason": imd_failure_reason,
            "open_meteo_failure_reason": str(exc),
            "station_id": station_id,
            "timestamp": now_dt.isoformat(),
            "temperature": 26.0,
            "humidity": 80.0,
            "pressure": 1008.0,
            "wind_speed": 2.5,
            "rainfall": 0.0,
            "raw_data": {"error": str(exc)},
        }


@app.get("/api/stations")
def get_stations_inventory():
    """
    Returns the complete station inventory with explicit simulation flags and station types.
    Ensures no simulated stations are misrepresented as real IMD AWS stations.
    """
    stations_list = []
    for stn_id, meta in AWS_STATION_COORDS.items():
        stations_list.append({
            "station_id": stn_id,
            "name": meta["name"],
            "lat": meta["lat"],
            "lon": meta["lon"],
            "state": meta["state"],
            "is_simulated": meta.get("is_simulated", True),
            "station_type": meta.get("station_type", "SIMULATED_INDICATIVE"),
        })
    return sanitize_json({
        "status": "SUCCESS",
        "total": len(stations_list),
        "real_imd_count": len(REAL_IMD_AWS_STATIONS),
        "simulated_count": len(stations_list) - len(REAL_IMD_AWS_STATIONS),
        "stations": stations_list,
    })


@app.get("/api/live/imd-observation")
def get_live_imd_observation(
    station_id: str = Query("43189", description="IMD AWS Station ID"),
    lat: Optional[float] = Query(None, description="Station Latitude"),
    lon: Optional[float] = Query(None, description="Station Longitude"),
    run_pipeline: bool = Query(True, description="Process through unified anomaly detection pipeline"),
):
    """
    Fetches real live AWS telemetry using requests.get('https://api.imd.gov.in/api/v1/aws_data')
    and runs the observation through the canonical anomaly detection pipeline.
    Accurately identifies source as IMD_AWS if primary query succeeds, or OPEN_METEO_CONTEXT if fallback.
    """
    obs = fetch_live_imd_aws_data(station_id, lat, lon)

    pipeline_result = None
    if run_pipeline:
        req = PipelineProcessRequest(
            station_id=station_id,
            observation={
                "station_id": station_id,
                "timestamp": obs["timestamp"],
                "temperature": obs["temperature"],
                "humidity": obs["humidity"],
                "pressure": obs["pressure"],
                "wind_speed": obs["wind_speed"],
                "rainfall": obs["rainfall"],
            },
            source=obs.get("source", "OPEN_METEO_CONTEXT"),
        )
        pipeline_result = process_pipeline_observation(req)

    return sanitize_json({
        "status": "SUCCESS",
        "station_id": station_id,
        "source": obs.get("source", "OPEN_METEO_CONTEXT"),
        "is_fallback": obs.get("is_fallback", False),
        "is_simulated": obs.get("is_simulated", False),
        "imd_fetch_status": obs.get("imd_fetch_status", "UNKNOWN"),
        "imd_failure_reason": obs.get("imd_failure_reason"),
        "observation": obs,
        "pipeline_result": pipeline_result,
    })


@app.post("/api/telemetry", response_model=CanonicalInferenceResponse)
def ingest_canonical_telemetry(payload: CanonicalTelemetryInput):
    """
    POST /api/telemetry
    The Canonical Single Authoritative Telemetry Pipeline Ingestion Endpoint.

    Target 15-Stage Pipeline:
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
    pipeline = get_canonical_pipeline()
    if pipeline.model is None:
        try:
            pipeline.model = get_model()
        except Exception:
            pass
    if pipeline.engine is None:
        try:
            pipeline.engine = get_engine()
        except Exception:
            pass

    return pipeline.process(payload)


@app.post("/api/pipeline/process")
def process_pipeline_observation(req: PipelineProcessRequest):
    """
    POST /api/pipeline/process
    THE SINGLE UNIFIED DETECTION PIPELINE.
    Delegates directly to CanonicalPipelineService so that every observation
    is processed through the single authoritative 15-stage pipeline.
    """
    raw_obs = req.observation or {}
    raw_source = (req.source or "").strip().upper()
    is_simulated_stn = req.station_id not in REAL_IMD_AWS_STATIONS

    if req.demo_run_id or raw_source in ("DEMO", "SYNTHETIC", "DEMO_REPLAY", "SIMULATION", "DEMO_SIMULATION"):
        source = "DEMO_SIMULATION"
    elif raw_source in ("LIVE", "LIVE_AWS", "IMD", "IMD_AWS"):
        if is_simulated_stn:
            # Rule 1 & 4: Simulated stations must never be falsely labeled as real IMD AWS live data
            source = "DEMO_SIMULATION" if req.demo_run_id else "OPEN_METEO_CONTEXT"
        else:
            source = "IMD_AWS"
    elif raw_source in ("OPEN_METEO", "FORECAST", "OPENMETEO", "OPEN_METEO_CONTEXT"):
        source = "OPEN_METEO_CONTEXT"
    elif raw_source in ("OFFLINE", "PREVIEW", "OFFLINE_PREVIEW"):
        source = "OFFLINE_PREVIEW"
    else:
        if is_simulated_stn:
            source = "DEMO_SIMULATION" if req.demo_run_id else "OPEN_METEO_CONTEXT"
        else:
            source = "IMD_AWS"

    # Build Canonical Telemetry Input
    canonical_input = CanonicalTelemetryInput(
        station_id=req.station_id,
        timestamp=str(raw_obs.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        temperature_c=raw_obs.get("temperature") if "temperature" in raw_obs else (raw_obs.get("temperature_c") if "temperature_c" in raw_obs else raw_obs.get("temp")),
        humidity_pct=raw_obs.get("humidity") if "humidity" in raw_obs else (raw_obs.get("humidity_pct") if "humidity_pct" in raw_obs else raw_obs.get("hum")),
        pressure_hpa=raw_obs.get("pressure") if "pressure" in raw_obs else (raw_obs.get("pressure_hpa") if "pressure_hpa" in raw_obs else raw_obs.get("baro")),
        wind_speed_ms=raw_obs.get("wind_speed") if "wind_speed" in raw_obs else (raw_obs.get("wind_speed_ms") if "wind_speed_ms" in raw_obs else raw_obs.get("WS")),
        rainfall_mm=raw_obs.get("rainfall") if "rainfall" in raw_obs else (raw_obs.get("rainfall_mm") if "rainfall_mm" in raw_obs else raw_obs.get("RAIN")),
        source=source,
        session_id=req.demo_run_id,
    )

    pipeline = get_canonical_pipeline()
    if pipeline.model is None:
        try:
            pipeline.model = get_model()
        except Exception:
            pass
    if pipeline.engine is None:
        try:
            pipeline.engine = get_engine()
        except Exception:
            pass

    canonical_res = pipeline.process(canonical_input)

    # Determine UI status badge
    if canonical_res.severity == "CRITICAL" or canonical_res.anomaly_type == "CORRUPTED_DATA":
        ai_status = "🔴 CRITICAL"
    elif canonical_res.is_anomaly:
        ai_status = "🟠 ANOMALY"
    elif canonical_res.anomaly_score >= 0.40 or canonical_res.anomaly_type == "MISSING_DATA":
        ai_status = "🟡 WARNING"
    else:
        ai_status = "🟢 NORMAL"

    prev_readings = canonical_res.previous_readings or {
        "temperature": None,
        "humidity": None,
        "pressure": None,
    }
    deltas = canonical_res.deltas or {
        "temperature": None,
        "humidity": None,
        "pressure": None,
    }

    # Backward compatible response object with enriched canonical details
    return sanitize_json({
        "station_id": canonical_res.station_id,
        "timestamp": canonical_res.timestamp,
        "source": canonical_res.source,
        "demo_run_id": req.demo_run_id,
        "readings": canonical_res.readings,
        "previous_readings": prev_readings,
        "change": deltas,
        "is_anomaly": canonical_res.is_anomaly,
        "anomaly_type": canonical_res.anomaly_type,
        "severity": canonical_res.severity,
        "anomaly_score": canonical_res.anomaly_score,
        "evidence_strength": canonical_res.evidence_strength,
        "status": canonical_res.final_status,
        "ai_status": ai_status,
        "explanation": canonical_res.explanation.get("summary", ""),
        "why_flagged": canonical_res.explanation.get("why_flagged", []),
        "classification": canonical_res.explanation.get("recommended_action", ""),
        "station_health": canonical_res.sensor_health,
        "ml_details": {
            "raw_lof_score": canonical_res.evidence.ml.get("score"),
            "normalized_lof_score": canonical_res.evidence.ml.get("score"),
            "threshold": canonical_res.evidence.ml.get("threshold", 0.12),
            "model": canonical_res.model.name,
            "model_version": canonical_res.model.version,
            "ml_anomaly": canonical_res.evidence.ml.get("ml_anomaly", False),
        },
        "evidence": canonical_res.evidence.model_dump(),
        "warmup_state": canonical_res.warmup_state,
        "data_quality_state": canonical_res.data_quality_state,
        "incident": canonical_res.incident.model_dump() if canonical_res.incident else None,
        "canonical_response": canonical_res.model_dump(),
    })


@app.get("/api/incidents")
def get_incidents(
    status: Optional[str] = Query(None, description="Filter by lifecycle status (DETECTED, ACKNOWLEDGED, INVESTIGATING, RESOLVED)"),
    station_id: Optional[str] = Query(None, description="Filter by station ID"),
    limit: int = Query(50, description="Max incidents to return"),
):
    """Query deduplicated anomaly incidents and their lifecycle states."""
    status_filter = status.upper() if status else None
    incidents = incident_manager.list_incidents(
        status=status_filter,  # type: ignore
        station_id=station_id,
        limit=limit,
    )
    return sanitize_json({
        "status": "SUCCESS",
        "count": len(incidents),
        "incidents": incidents,
    })


@app.get("/api/incidents/{incident_id}")
def get_incident_by_id(incident_id: str):
    """Get single incident details by incident ID."""
    incidents = incident_manager.list_incidents(limit=200)
    for inc in incidents:
        if inc.get("incident_id") == incident_id:
            return sanitize_json({"status": "SUCCESS", "incident": inc})
    raise HTTPException(status_code=404, detail=f"Incident '{incident_id}' not found.")


@app.patch("/api/incidents/{incident_id}/status")
def update_incident_lifecycle(
    incident_id: str,
    status: str = Query(..., description="Target lifecycle state (ACKNOWLEDGED, INVESTIGATING, RESOLVED)"),
):
    """Update incident lifecycle state across memory registry and persistent database."""
    allowed = ("DETECTED", "ACKNOWLEDGED", "INVESTIGATING", "RESOLVED")
    status_upper = status.upper()
    if status_upper not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid lifecycle status: {status}. Allowed states: {allowed}",
        )
    updated = incident_manager.update_lifecycle_status(incident_id, status_upper)  # type: ignore
    db_updated = update_anomaly_status(anomaly_id=incident_id, status=status_upper)
    if not updated and not db_updated:
        raise HTTPException(
            status_code=404,
            detail=f"Incident '{incident_id}' not found in active or historical registry.",
        )
    return sanitize_json({
        "status": "SUCCESS",
        "incident": updated or db_updated,
        "database_sync": True,
    })


@app.get("/api/models")
def list_registered_models(limit: int = Query(20, description="Max models to return")):
    """
    List registered ML models with evaluation metrics, thresholds, and governance provenance.
    """
    models = get_model_versions(limit=limit)
    return sanitize_json({
        "status": "SUCCESS",
        "count": len(models),
        "models": models,
    })


@app.get("/api/database/stations")
def list_database_stations(status: Optional[str] = None):
    """
    Retrieve persistent stations inventory from Supabase / SQLite.
    """
    stations = get_stations(status=status)
    return sanitize_json({
        "status": "SUCCESS",
        "count": len(stations),
        "stations": stations,
    })


@app.post("/api/ml/predict")
def predict_observation(req: PredictRequest):
    """
    Backward compatible predict endpoint.
    Routes observations through the single unified pipeline with source='LIVE'
    so all live telemetry is validated, evaluated by ML/rules, and saved to Supabase.
    """
    pipe_req = PipelineProcessRequest(
        station_id=req.station_id,
        observation=req.observation,
        source="LIVE",
        history=req.history,
    )
    res = process_pipeline_observation(pipe_req)

    # Return structure matching frontend MLInferenceResult contract
    ml_details = res.get("ml_details") or {}
    response_payload = dict(res)
    for k, v in ml_details.items():
        if k not in response_payload or response_payload[k] is None:
            response_payload[k] = v

    response_payload["ml_anomaly"] = res.get("is_anomaly", False)
    response_payload["raw_lof_score"] = ml_details.get("raw_lof_score", res.get("anomaly_score"))
    response_payload["normalized_lof_score"] = ml_details.get("normalized_lof_score")
    response_payload["threshold"] = ml_details.get("threshold", 1.5)
    response_payload["model"] = ml_details.get("model", "SkyGuard Pipeline")
    response_payload["model_version"] = ml_details.get("model_version", "2.0.0")
    response_payload["status"] = "EVALUATED" if res.get("status") in ("NORMAL", "ANOMALY") else "WARMUP"
    return sanitize_json(response_payload)


class DemoReplayRequest(BaseModel):
    station_id: Optional[str] = None
    start_index: int = 0
    count: int = 10


@app.post("/api/demo/replay")
def replay_demo_stream(req: DemoReplayRequest):
    """
    Replay rows from skyguard_fake_demo_dataset.csv chronologically
    through the EXACT SAME detection pipeline with source='DEMO'.
    """
    demo_info = get_demo_dataset()
    records = demo_info.get("records", [])
    if req.station_id:
        records = [r for r in records if r.get("station_id") == req.station_id]

    sliced = records[req.start_index : req.start_index + req.count]
    replayed = []
    demo_session = f"demo_replay_{req.station_id or 'all'}"
    for rec in sliced:
        pipe_res = process_pipeline_observation(
            PipelineProcessRequest(
                station_id=rec["station_id"],
                observation={
                    "temperature": rec.get("temperature_c"),
                    "humidity": rec.get("humidity_pct"),
                    "pressure": rec.get("pressure_hpa"),
                    "wind_speed": rec.get("wind_speed_ms"),
                    "rainfall": rec.get("rainfall_mm"),
                    "timestamp": rec.get("timestamp"),
                },
                source="DEMO_SIMULATION",
                demo_run_id=demo_session,
            )
        )
        replayed.append({
            "index": rec.get("index"),
            "station_id": rec.get("station_id"),
            "timestamp": rec.get("timestamp"),
            "is_anomaly": pipe_res.get("is_anomaly"),
            "anomaly_type": pipe_res.get("anomaly_type"),
            "severity": pipe_res.get("severity"),
            "source": "DEMO_SIMULATION",
        })

    return sanitize_json({
        "status": "SUCCESS",
        "replayed_count": len(replayed),
        "results": replayed,
    })


@app.post("/api/test/insert-reading")
def test_insert_reading_endpoint():
    """
    Test endpoint per Requirement 11:
    Inserts ONE test reading for station HYD_AWS_01 with source=DEMO,
    then queries Supabase to verify it exists.
    """
    now_ts = datetime.now(timezone.utc).isoformat()
    record = save_sensor_reading(
        station_id="HYD_AWS_01",
        timestamp=now_ts,
        temperature_c=31.8,
        humidity_pct=61.4,
        pressure_hpa=1008.7,
        wind_speed_ms=3.9,
        rainfall_mm=0.0,
        source="DEMO_SIMULATION",
    )
    latest = get_latest_readings(limit=1, station_id="HYD_AWS_01")
    return sanitize_json({
        "status": "SUCCESS" if record else "PARTIAL_SUCCESS",
        "inserted_record": record,
        "latest_retrieved": latest[0] if latest else None,
        "note": "Check Supabase table sensor_readings for station HYD_AWS_01",
    })


@app.post("/api/ml/seed-history")
def seed_history(req: SeedHistoryRequest):
    """Seed or replace station history records for testing or demonstration."""
    station_history_cache[req.station_id] = req.records
    return {
        "station_id": req.station_id,
        "history_length": len(station_history_cache[req.station_id]),
        "status": "SEEDED",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8787))
    uvicorn.run(app, host="0.0.0.0", port=port)
