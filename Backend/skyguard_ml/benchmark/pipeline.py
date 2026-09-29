"""
SkyGuard AI — Master Benchmark Dataset Pipeline.

Orchestrates the creation, chronological splitting, scenario injection,
and persistence of the SkyGuard benchmark dataset across:
datasets/
├── raw/
├── processed/
├── clean/
├── synthetic_faults/
├── weather_events/
└── benchmark/
"""

from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from .config import (
    BenchmarkConfig,
    StationConfig,
    STATIONS,
    FAULT_TYPES,
    BENCHMARK_COLUMNS,
)
from .clean_generator import generate_clean_network_telemetry
from .fault_generator import inject_fault_into_slice
from .weather_generator import (
    inject_convective_storm_front,
    inject_regional_heatwave,
    inject_synoptic_low_pressure,
    inject_coastal_sea_breeze,
)
from .multi_station import inject_case_a_isolated_fault, inject_case_b_coherent_event


class BenchmarkPipeline:
    """
    Constructs the end-to-end reproducible benchmark dataset.
    """

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        self.config = config or BenchmarkConfig()
        self.rng = np.random.default_rng(self.config.random_seed)

    def _ensure_directories(self) -> None:
        """Create target dataset directories without overwriting existing files."""
        for directory in (
            self.config.raw_dir,
            self.config.processed_dir,
            self.config.clean_dir,
            self.config.synthetic_faults_dir,
            self.config.weather_events_dir,
            self.config.benchmark_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def generate_station_metadata(self) -> pd.DataFrame:
        """Write station metadata to datasets/raw/station_metadata.csv."""
        rows = [
            {
                "station_id": s.station_id,
                "station_name": s.station_name,
                "state": s.state,
                "district": s.district,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "elevation_m": s.elevation_m,
                "base_temp_c": s.base_temp,
                "base_hum_pct": s.base_hum,
                "base_baro_hpa": s.base_baro,
                "cluster_id": s.cluster_id,
            }
            for s in STATIONS
        ]
        frame = pd.DataFrame(rows)
        dest = self.config.raw_dir / "station_metadata.csv"
        frame.to_csv(dest, index=False)
        return frame

    def generate_clean_data(self) -> pd.DataFrame:
        """Generate clean baseline observations and persist to datasets/clean/clean_telemetry.csv."""
        clean_df = generate_clean_network_telemetry(self.config, STATIONS)
        dest = self.config.clean_dir / "clean_telemetry.csv"
        clean_df.to_csv(dest, index=False)

        # Also calculate and persist summary baseline statistics to datasets/processed/
        baselines = clean_df.groupby("station_id").agg({
            "temperature": ["mean", "std", "min", "max"],
            "humidity": ["mean", "std", "min", "max"],
            "pressure": ["mean", "std", "min", "max"],
        }).reset_index()
        baselines.columns = [
            "_".join(col).strip("_") for col in baselines.columns.values
        ]
        baselines.to_csv(self.config.processed_dir / "station_baselines.csv", index=False)

        return clean_df

    def build_benchmark_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build full benchmark dataset with chronological splits:
        - Train (Days 1 to 27): 100% clean normal data.
        - Validation (Days 28 to 36): Clean background + validation calibration scenarios.
        - Test (Days 37 to 45): Clean background + independent test benchmark scenarios.
        """
        self._ensure_directories()
        self.generate_station_metadata()

        clean_df = self.generate_clean_data()
        records_by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for rec in clean_df.to_dict(orient="records"):
            records_by_station[rec["station_id"]].append(rec)

        total_hours = self.config.total_hours
        train_end = int(total_hours * self.config.train_fraction)  # index 648 (Day 27 end)
        val_end = int(total_hours * (self.config.train_fraction + self.config.val_fraction))  # index 864 (Day 36 end)

        # -------------------------------------------------------------
        # 1. INJECT VALIDATION SCENARIOS (strictly within index 648 to 863)
        # -------------------------------------------------------------
        # Val Window: hour 648 to 863 (duration 216 hours)
        # Station 43189 (Vijayawada):
        # Val Fault 1: MILD temperature drift (hours 660-668)
        s_idx, dur = 660, 8
        records_by_station["43189"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43189"][s_idx : s_idx + dur],
            fault_type="TEMPERATURE_DRIFT",
            severity="MILD",
            scenario_id="SCN_VAL_DRIFT_43189",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Val Fault 2: MODERATE temperature spike (hours 700-704)
        s_idx, dur = 700, 4
        records_by_station["43189"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43189"][s_idx : s_idx + dur],
            fault_type="TEMPERATURE_SPIKE",
            severity="MODERATE",
            scenario_id="SCN_VAL_SPIKE_43189",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Station AWS002 (Guntur):
        # Val Fault 3: MODERATE frozen sensor (hours 720-728)
        s_idx, dur = 720, 8
        records_by_station["AWS002"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["AWS002"][s_idx : s_idx + dur],
            fault_type="FROZEN_SENSOR",
            severity="MODERATE",
            scenario_id="SCN_VAL_FROZEN_AWS002",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Station AWS004 (Machilipatnam):
        # Val Fault 4: MILD humidity spike (hours 740-746)
        s_idx, dur = 740, 6
        records_by_station["AWS004"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["AWS004"][s_idx : s_idx + dur],
            fault_type="HUMIDITY_SPIKE",
            severity="MILD",
            scenario_id="SCN_VAL_HUM_SPIKE_AWS004",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Station 43150 (Visakhapatnam):
        # Val Fault 5: MODERATE pressure drift (hours 760-770)
        s_idx, dur = 760, 10
        records_by_station["43150"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43150"][s_idx : s_idx + dur],
            fault_type="PRESSURE_DRIFT",
            severity="MODERATE",
            scenario_id="SCN_VAL_BARO_DRIFT_43150",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Station 43245 (Tirupati):
        # Val Fault 6: SEVERE missing data (hours 780-792)
        s_idx, dur = 780, 12
        records_by_station["43245"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43245"][s_idx : s_idx + dur],
            fault_type="MISSING_DATA",
            severity="SEVERE",
            scenario_id="SCN_VAL_MISSING_43245",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Station HYD_AWS_01 (Hyderabad):
        # Val Fault 7: MILD multivariate inconsistency (hours 800-806)
        s_idx, dur = 800, 6
        records_by_station["HYD_AWS_01"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["HYD_AWS_01"][s_idx : s_idx + dur],
            fault_type="MULTIVARIATE_INCONSISTENCY",
            severity="MILD",
            scenario_id="SCN_VAL_MULTI_HYD",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Val Weather Event 1: Convective storm front across delta cluster (hours 820-825)
        inject_convective_storm_front(
            records_by_station=records_by_station,
            station_ids=["43189", "AWS002", "AWS004"],
            start_index=820,
            duration_hours=5,
            scenario_id="SCN_VAL_WX_CONVECTIVE_STORM",
            propagation_lags={"43189": 0, "AWS002": 1, "AWS004": 1},
        )

        # Val Weather Event 2: Coastal sea breeze front at Visakhapatnam & Machilipatnam (hours 840-846)
        inject_coastal_sea_breeze(
            records_by_station=records_by_station,
            station_ids=["43150", "AWS004"],
            start_index=840,
            duration_hours=6,
            scenario_id="SCN_VAL_WX_SEA_BREEZE",
        )

        # -------------------------------------------------------------
        # 2. INJECT TEST SCENARIOS (strictly within index 864 to 1079)
        # -------------------------------------------------------------
        # Test Window: hour 864 to 1079 (duration 216 hours)

        # CASE A: Isolated station sensor fault (Vijayawada 43189 drifts severely, Guntur & Machilipatnam stay normal)
        inject_case_a_isolated_fault(
            records_by_station=records_by_station,
            target_station_id="43189",
            peer_station_ids=["AWS002", "AWS004"],
            start_index=880,
            duration_hours=10,
            fault_type="TEMPERATURE_DRIFT",
            severity="SEVERE",
            scenario_id="SCN_TEST_CASE_A_ISOLATED_DRIFT_43189",
            rng=self.rng,
        )

        # CASE B: Coherent regional storm (Vijayawada 43189, Guntur AWS002, Machilipatnam AWS004 storm passage)
        inject_case_b_coherent_event(
            records_by_station=records_by_station,
            station_ids=["43189", "AWS002", "AWS004"],
            start_index=910,
            duration_hours=5,
            scenario_id="SCN_TEST_CASE_B_COHERENT_STORM_DELTA",
        )

        # Test Fault: Temperature Spike across severities on 43150 (Visakhapatnam)
        # Mild spike (hours 930-933)
        s_idx, dur = 930, 3
        records_by_station["43150"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43150"][s_idx : s_idx + dur],
            fault_type="TEMPERATURE_SPIKE",
            severity="MILD",
            scenario_id="SCN_TEST_SPIKE_MILD_43150",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Temperature Drop on 43245 (Tirupati) (hours 940-944)
        s_idx, dur = 940, 4
        records_by_station["43245"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43245"][s_idx : s_idx + dur],
            fault_type="TEMPERATURE_DROP",
            severity="MODERATE",
            scenario_id="SCN_TEST_DROP_MOD_43245",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Humidity Drift on HYD_AWS_01 (hours 955-965)
        s_idx, dur = 955, 10
        records_by_station["HYD_AWS_01"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["HYD_AWS_01"][s_idx : s_idx + dur],
            fault_type="HUMIDITY_DRIFT",
            severity="MODERATE",
            scenario_id="SCN_TEST_HUM_DRIFT_HYD",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Pressure Spike on AWS002 (hours 970-974)
        s_idx, dur = 970, 4
        records_by_station["AWS002"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["AWS002"][s_idx : s_idx + dur],
            fault_type="PRESSURE_SPIKE",
            severity="SEVERE",
            scenario_id="SCN_TEST_BARO_SPIKE_AWS002",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Frozen Sensor on AWS004 (hours 985-995)
        s_idx, dur = 985, 10
        records_by_station["AWS004"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["AWS004"][s_idx : s_idx + dur],
            fault_type="FROZEN_SENSOR",
            severity="SEVERE",
            scenario_id="SCN_TEST_FROZEN_AWS004",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Duplicate Data on 43189 (hours 1000-1006)
        s_idx, dur = 1000, 6
        records_by_station["43189"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43189"][s_idx : s_idx + dur],
            fault_type="DUPLICATE_DATA",
            severity="MODERATE",
            scenario_id="SCN_TEST_DUPLICATE_43189",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Timestamp Error on 43150 (hours 1015-1018)
        s_idx, dur = 1015, 3
        records_by_station["43150"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43150"][s_idx : s_idx + dur],
            fault_type="TIMESTAMP_ERROR",
            severity="MODERATE",
            scenario_id="SCN_TEST_TIMESTAMP_43150",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Communication Failure on 43245 (hours 1025-1033)
        s_idx, dur = 1025, 8
        records_by_station["43245"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["43245"][s_idx : s_idx + dur],
            fault_type="COMMUNICATION_FAILURE",
            severity="MODERATE",
            scenario_id="SCN_TEST_COMM_FAIL_43245",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Fault: Multivariate Inconsistency on HYD_AWS_01 (hours 1038-1044)
        s_idx, dur = 1038, 6
        records_by_station["HYD_AWS_01"][s_idx : s_idx + dur] = inject_fault_into_slice(
            records_by_station["HYD_AWS_01"][s_idx : s_idx + dur],
            fault_type="MULTIVARIATE_INCONSISTENCY",
            severity="MODERATE",
            scenario_id="SCN_TEST_MULTI_HYD",
            rng=self.rng,
            start_offset=0,
            duration=dur,
        )

        # Test Weather Event 3: Regional Heatwave Surge across inland stations (hours 1045-1057)
        inject_regional_heatwave(
            records_by_station=records_by_station,
            station_ids=["HYD_AWS_01", "43245", "AWS002"],
            start_index=1045,
            duration_hours=12,
            scenario_id="SCN_TEST_WX_HEATWAVE",
        )

        # Test Weather Event 4: Synoptic Low Pressure Depression across all 6 stations (hours 1058-1078)
        inject_synoptic_low_pressure(
            records_by_station=records_by_station,
            station_ids=[s.station_id for s in STATIONS],
            start_index=1058,
            duration_hours=20,
            scenario_id="SCN_TEST_WX_SYNOPTIC_DEPRESSION",
        )

        # -------------------------------------------------------------
        # 3. ASSEMBLE COMBINED BENCHMARK AND SPLITS
        # -------------------------------------------------------------
        all_benchmark_records: list[dict[str, Any]] = []
        train_records: list[dict[str, Any]] = []
        val_records: list[dict[str, Any]] = []
        test_records: list[dict[str, Any]] = []
        fault_records_only: list[dict[str, Any]] = []
        weather_records_only: list[dict[str, Any]] = []

        for stn_id, stn_records in records_by_station.items():
            for idx, r in enumerate(stn_records):
                all_benchmark_records.append(r)

                if idx < train_end:
                    train_records.append(r)
                elif idx < val_end:
                    val_records.append(r)
                else:
                    test_records.append(r)

                if r["ground_truth"] in ("SENSOR_FAULT", "DATA_QUALITY_ISSUE", "COMMUNICATION_FAULT"):
                    fault_records_only.append(r)
                elif r["ground_truth"] == "WEATHER_EVENT":
                    weather_records_only.append(r)

        full_df = pd.DataFrame(all_benchmark_records, columns=BENCHMARK_COLUMNS)
        train_df = pd.DataFrame(train_records, columns=BENCHMARK_COLUMNS)
        val_df = pd.DataFrame(val_records, columns=BENCHMARK_COLUMNS)
        test_df = pd.DataFrame(test_records, columns=BENCHMARK_COLUMNS)

        # Sort all dataframes deterministically
        for df in (full_df, train_df, val_df, test_df):
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], utc=True)
            df.sort_values(["station_id", "timestamp_dt"], inplace=True)
            df.drop(columns=["timestamp_dt"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        # Save partitioned files
        full_df.to_csv(self.config.benchmark_dir / "benchmark_dataset.csv", index=False)
        train_df.to_csv(self.config.benchmark_dir / "benchmark_train.csv", index=False)
        val_df.to_csv(self.config.benchmark_dir / "benchmark_val.csv", index=False)
        test_df.to_csv(self.config.benchmark_dir / "benchmark_test.csv", index=False)

        # Save synthetic faults and weather events subsets
        if fault_records_only:
            fault_df = pd.DataFrame(fault_records_only, columns=BENCHMARK_COLUMNS)
            fault_df.to_csv(self.config.synthetic_faults_dir / "synthetic_faults.csv", index=False)
        if weather_records_only:
            wx_df = pd.DataFrame(weather_records_only, columns=BENCHMARK_COLUMNS)
            wx_df.to_csv(self.config.weather_events_dir / "weather_events.csv", index=False)

        # Generate metadata report
        metadata = self._build_metadata_summary(full_df, train_df, val_df, test_df)
        with open(self.config.benchmark_dir / "benchmark_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return full_df, train_df, val_df, test_df

    def _build_metadata_summary(
        self,
        full_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate comprehensive metadata summary dictionary."""
        fault_type_counts = full_df[full_df["fault_type"] != "NONE"]["fault_type"].value_counts().to_dict()
        severity_counts = full_df[full_df["severity"] != "NONE"]["severity"].value_counts().to_dict()
        station_counts = full_df["station_id"].value_counts().to_dict()
        ground_truth_counts = full_df["ground_truth"].value_counts().to_dict()

        return {
            "benchmark_name": "SkyGuard-Scientific-Benchmark-v1.0",
            "random_seed": self.config.random_seed,
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "station_count": int(full_df["station_id"].nunique()),
            "total_observations": int(len(full_df)),
            "time_range": {
                "start": str(full_df["timestamp"].min()),
                "end": str(full_df["timestamp"].max()),
                "total_hours": self.config.total_hours,
                "cadence_hours": self.config.cadence_hours,
            },
            "split_summary": {
                "train": {
                    "count": int(len(train_df)),
                    "percentage": round(len(train_df) / len(full_df) * 100, 2),
                    "start": str(train_df["timestamp"].min()),
                    "end": str(train_df["timestamp"].max()),
                    "clean_normal_only": bool((train_df["ground_truth"] == "NORMAL").all()),
                },
                "validation": {
                    "count": int(len(val_df)),
                    "percentage": round(len(val_df) / len(full_df) * 100, 2),
                    "start": str(val_df["timestamp"].min()),
                    "end": str(val_df["timestamp"].max()),
                    "ground_truth_distribution": val_df["ground_truth"].value_counts().to_dict(),
                },
                "test": {
                    "count": int(len(test_df)),
                    "percentage": round(len(test_df) / len(full_df) * 100, 2),
                    "start": str(test_df["timestamp"].min()),
                    "end": str(test_df["timestamp"].max()),
                    "ground_truth_distribution": test_df["ground_truth"].value_counts().to_dict(),
                },
            },
            "class_distribution": ground_truth_counts,
            "fault_type_distribution": fault_type_counts,
            "severity_distribution": severity_counts,
            "station_distribution": station_counts,
            "scenario_counts": int(full_df["scenario_id"].nunique()),
            "distinct_scenarios": sorted(full_df["scenario_id"].unique().tolist()),
            "limitations_and_notes": [
                "Baseline observations are physically calibrated synthetic simulations mimicking Indian Automatic Weather Station diurnal curves, not raw IMD telemetric recordings.",
                "Weather events are physically consistent meteorological simulations designed to benchmark spatial evidence fusion against false positive sensor alerts.",
                "Training split is guaranteed strictly 100% clean normal observations with zero fault contamination.",
            ],
        }
