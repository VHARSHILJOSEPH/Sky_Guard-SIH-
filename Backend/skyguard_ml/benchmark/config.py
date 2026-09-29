"""
SkyGuard AI — Benchmark Configuration & Schema Definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ============================================================
# Ground Truth Taxonomy
# ============================================================

GroundTruthType = Literal[
    "NORMAL",
    "SENSOR_FAULT",
    "DATA_QUALITY_ISSUE",
    "COMMUNICATION_FAULT",
    "WEATHER_EVENT",
]

SeverityLevel = Literal[
    "NONE",
    "MILD",
    "MODERATE",
    "SEVERE",
]

ScenarioType = Literal[
    "CLEAN_NORMAL",
    "SINGLE_STATION_FAULT",
    "MULTI_STATION_WEATHER_EVENT",
    "DATA_QUALITY",
    "COMMUNICATION",
]

# Supported Fault Types matching SkyGuard architecture
FAULT_TYPES = (
    "TEMPERATURE_SPIKE",
    "TEMPERATURE_DROP",
    "TEMPERATURE_DRIFT",
    "HUMIDITY_SPIKE",
    "HUMIDITY_DRIFT",
    "PRESSURE_SPIKE",
    "PRESSURE_DRIFT",
    "FROZEN_SENSOR",
    "MISSING_DATA",
    "DUPLICATE_DATA",
    "TIMESTAMP_ERROR",
    "COMMUNICATION_FAILURE",
    "MULTIVARIATE_INCONSISTENCY",
)

WEATHER_EVENT_TYPES = (
    "CONVECTIVE_STORM_FRONT",
    "REGIONAL_HEATWAVE_SURGE",
    "SYNOPTIC_LOW_PRESSURE_TROUGH",
    "COASTAL_SEA_BREEZE_INTRUSION",
)

# Standard observation columns
BASE_PARAMETERS = ("temperature", "humidity", "pressure")

BENCHMARK_COLUMNS = [
    "station_id",
    "station_name",
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "source",
    "scenario_id",
    "scenario_type",
    "fault_type",
    "severity",
    "ground_truth",
    "is_anomaly",
    "is_sensor_fault",
    "event_start",
    "event_end",
    "affected_parameter",
    "description",
]


# ============================================================
# Station Definitions
# ============================================================

@dataclass(frozen=True)
class StationConfig:
    station_id: str
    station_name: str
    state: str
    district: str
    latitude: float
    longitude: float
    elevation_m: float
    base_temp: float
    base_hum: float
    base_baro: float
    cluster_id: str


# Network of 6 stations covering a clustered delta region + reference stations
STATIONS = [
    # Cluster 1: Krishna-Guntur Delta Cluster (tightly coupled for spatial checks, ~30-60 km apart)
    StationConfig(
        station_id="43189",
        station_name="Vijayawada (AWS014)",
        state="Andhra Pradesh",
        district="NTR",
        latitude=16.5062,
        longitude=80.6480,
        elevation_m=25.0,
        base_temp=32.4,
        base_hum=64.0,
        base_baro=1008.2,
        cluster_id="DELTA_CLUSTER",
    ),
    StationConfig(
        station_id="AWS002",
        station_name="Guntur (AWS002)",
        state="Andhra Pradesh",
        district="Guntur",
        latitude=16.3067,
        longitude=80.4365,
        elevation_m=33.0,
        base_temp=32.1,
        base_hum=63.5,
        base_baro=1007.8,
        cluster_id="DELTA_CLUSTER",
    ),
    StationConfig(
        station_id="AWS004",
        station_name="Machilipatnam (AWS004)",
        state="Andhra Pradesh",
        district="Krishna",
        latitude=16.1875,
        longitude=81.1389,
        elevation_m=4.0,
        base_temp=30.5,
        base_hum=75.0,
        base_baro=1010.5,
        cluster_id="DELTA_CLUSTER",
    ),
    # Reference Station 2: Northern Coastal Station
    StationConfig(
        station_id="43150",
        station_name="Visakhapatnam (AWS008)",
        state="Andhra Pradesh",
        district="Visakhapatnam",
        latitude=17.6868,
        longitude=83.2185,
        elevation_m=15.0,
        base_temp=29.8,
        base_hum=78.0,
        base_baro=1012.4,
        cluster_id="NORTH_COAST",
    ),
    # Reference Station 3: Southern Inland / Foothills Station
    StationConfig(
        station_id="43245",
        station_name="Tirupati (AWS021)",
        state="Andhra Pradesh",
        district="Tirupati",
        latitude=13.6288,
        longitude=79.4192,
        elevation_m=161.0,
        base_temp=34.1,
        base_hum=52.0,
        base_baro=1004.8,
        cluster_id="SOUTH_INLAND",
    ),
    # Reference Station 4: Deccan Plateau Urban Station
    StationConfig(
        station_id="HYD_AWS_01",
        station_name="Hyderabad (AWS030)",
        state="Telangana",
        district="Hyderabad",
        latitude=17.3850,
        longitude=78.4867,
        elevation_m=542.0,
        base_temp=31.8,
        base_hum=61.4,
        base_baro=1008.7,
        cluster_id="DECCAN_PLATEAU",
    ),
]


# ============================================================
# Benchmark Pipeline Configuration
# ============================================================

@dataclass
class BenchmarkConfig:
    # Deterministic master seed
    random_seed: int = 42

    # Timeline settings (45 days of hourly observations = 1,080 intervals per station)
    start_timestamp: str = "2026-08-01T00:00:00Z"
    total_hours: int = 1080  # 45 days
    cadence_hours: int = 1

    # Chronological partition ratios (60% Train, 20% Val, 20% Test)
    train_fraction: float = 0.60  # Days 1 to 27 (Hours 0 to 647)
    val_fraction: float = 0.20    # Days 28 to 36 (Hours 648 to 863)
    test_fraction: float = 0.20   # Days 37 to 45 (Hours 864 to 1079)

    # Base directory paths
    workspace_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent.parent)

    @property
    def datasets_dir(self) -> Path:
        return self.workspace_root / "datasets"

    @property
    def raw_dir(self) -> Path:
        return self.datasets_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.datasets_dir / "processed"

    @property
    def clean_dir(self) -> Path:
        return self.datasets_dir / "clean"

    @property
    def synthetic_faults_dir(self) -> Path:
        return self.datasets_dir / "synthetic_faults"

    @property
    def weather_events_dir(self) -> Path:
        return self.datasets_dir / "weather_events"

    @property
    def benchmark_dir(self) -> Path:
        return self.datasets_dir / "benchmark"
