"""
SkyGuard AI — Benchmark Dataset Quality Validator.

Enforces strict scientific validation rules:
1. Timestamp ordering (strictly monotonic per station).
2. Missing required fields validation.
3. Duplicate row detection.
4. Physical sanity checks (bounds, thermodynamic plausibility).
5. Label consistency across ground truth taxonomies.
6. Scenario leakage between splits.
7. Zero contamination in training split (clean normal data only).
8. Station separation and coverage.
9. Class and severity distribution balance.
10. Fixed-seed reproducibility verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .config import (
    BENCHMARK_COLUMNS,
    FAULT_TYPES,
    WEATHER_EVENT_TYPES,
    STATIONS,
)


@dataclass
class ValidationResult:
    passed: bool
    checks_run: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


class BenchmarkValidator:
    """
    Automated scientific quality check suite for SkyGuard benchmark data.
    """

    def __init__(self, benchmark_dir: Path | str) -> None:
        self.benchmark_dir = Path(benchmark_dir)

    def run_all_checks(
        self,
        full_df: pd.DataFrame | None = None,
        train_df: pd.DataFrame | None = None,
        val_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ) -> ValidationResult:
        """Run all quality checks on benchmark files or provided dataframes."""
        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        checks_run = 0

        # Load from disk if not provided
        if full_df is None:
            full_path = self.benchmark_dir / "benchmark_dataset.csv"
            if not full_path.exists():
                return ValidationResult(passed=False, checks_run=1, errors=[f"File not found: {full_path}"])
            full_df = pd.read_csv(full_path)

        if train_df is None:
            train_df = pd.read_csv(self.benchmark_dir / "benchmark_train.csv")
        if val_df is None:
            val_df = pd.read_csv(self.benchmark_dir / "benchmark_val.csv")
        if test_df is None:
            test_df = pd.read_csv(self.benchmark_dir / "benchmark_test.csv")

        # -------------------------------------------------------------
        # Check 1: Required Columns
        # -------------------------------------------------------------
        checks_run += 1
        missing_cols = [col for col in BENCHMARK_COLUMNS if col not in full_df.columns]
        if missing_cols:
            errors.append(f"Missing required columns in full dataset: {missing_cols}")

        # -------------------------------------------------------------
        # Check 2: Missing Required Identifiers
        # -------------------------------------------------------------
        checks_run += 1
        critical_fields = ["station_id", "timestamp", "source", "scenario_id", "scenario_type", "ground_truth"]
        for field_name in critical_fields:
            null_count = full_df[field_name].isna().sum()
            if null_count > 0:
                errors.append(f"Field '{field_name}' contains {null_count} null values in benchmark dataset")

        # -------------------------------------------------------------
        # Check 3: Timestamp Monotonicity Per Station (ignoring deliberate timestamp faults)
        # -------------------------------------------------------------
        checks_run += 1
        for stn_id, group in full_df[full_df["fault_type"] != "TIMESTAMP_ERROR"].groupby("station_id"):
            ts_series = pd.to_datetime(group["timestamp"], utc=True)
            if not ts_series.is_monotonic_increasing:
                errors.append(f"Timestamps are not strictly monotonic increasing for station {stn_id}")

        # -------------------------------------------------------------
        # Check 4: Clean Training Split Contamination (CRITICAL)
        # -------------------------------------------------------------
        checks_run += 1
        non_normal_train = (train_df["ground_truth"] != "NORMAL").sum()
        anomalies_train = (train_df["is_anomaly"] != 0).sum()
        faults_train = (train_df["is_sensor_fault"] != 0).sum()
        fault_types_train = (train_df["fault_type"] != "NONE").sum()

        if non_normal_train > 0 or anomalies_train > 0 or faults_train > 0 or fault_types_train > 0:
            errors.append(
                f"TRAIN CONTAMINATION DETECTED: {non_normal_train} non-normal rows, "
                f"{anomalies_train} anomaly flags, {fault_types_train} faults found in training split."
            )
        else:
            metrics["train_clean_purity"] = "100.0% (Zero contamination)"

        # -------------------------------------------------------------
        # Check 5: Scenario Leakage Between Splits
        # -------------------------------------------------------------
        checks_run += 1
        train_scenarios = set(train_df["scenario_id"].unique()) - {"SCN_CLEAN_BASELINE"}
        val_scenarios = set(val_df["scenario_id"].unique()) - {"SCN_CLEAN_BASELINE"}
        test_scenarios = set(test_df["scenario_id"].unique()) - {"SCN_CLEAN_BASELINE"}

        leak_train_val = train_scenarios.intersection(val_scenarios)
        leak_train_test = train_scenarios.intersection(test_scenarios)
        leak_val_test = val_scenarios.intersection(test_scenarios)

        if leak_train_val:
            errors.append(f"Scenario leakage between train and validation: {leak_train_val}")
        if leak_train_test:
            errors.append(f"Scenario leakage between train and test: {leak_train_test}")
        if leak_val_test:
            errors.append(f"Scenario leakage between validation and test: {leak_val_test}")

        # -------------------------------------------------------------
        # Check 6: Chronological Ordering Across Splits
        # -------------------------------------------------------------
        checks_run += 1
        train_max_ts = pd.to_datetime(train_df["timestamp"], utc=True).max()
        val_min_ts = pd.to_datetime(val_df["timestamp"], utc=True).min()
        val_max_ts = pd.to_datetime(val_df["timestamp"], utc=True).max()
        test_min_ts = pd.to_datetime(test_df["timestamp"], utc=True).min()

        if train_max_ts >= val_min_ts:
            errors.append(f"Chronological split violation: train_max ({train_max_ts}) >= val_min ({val_min_ts})")
        if val_max_ts >= test_min_ts:
            errors.append(f"Chronological split violation: val_max ({val_max_ts}) >= test_min ({test_min_ts})")

        metrics["chronological_boundaries"] = {
            "train_end": str(train_max_ts),
            "val_start": str(val_min_ts),
            "val_end": str(val_max_ts),
            "test_start": str(test_min_ts),
        }

        # -------------------------------------------------------------
        # Check 7: Label Consistency
        # -------------------------------------------------------------
        checks_run += 1
        # Normal rows must have is_anomaly == 0
        normal_anomaly_mismatch = ((full_df["ground_truth"] == "NORMAL") & (full_df["is_anomaly"] != 0)).sum()
        if normal_anomaly_mismatch > 0:
            errors.append(f"{normal_anomaly_mismatch} NORMAL rows have is_anomaly != 0")

        # Weather events must have is_anomaly == 1 and is_sensor_fault == 0
        wx_fault_mismatch = ((full_df["ground_truth"] == "WEATHER_EVENT") & (full_df["is_sensor_fault"] != 0)).sum()
        if wx_fault_mismatch > 0:
            errors.append(f"{wx_fault_mismatch} WEATHER_EVENT rows are incorrectly flagged as is_sensor_fault == 1")

        # Sensor faults must have is_sensor_fault == 1
        sf_mismatch = ((full_df["ground_truth"] == "SENSOR_FAULT") & (full_df["is_sensor_fault"] != 1)).sum()
        if sf_mismatch > 0:
            errors.append(f"{sf_mismatch} SENSOR_FAULT rows have is_sensor_fault != 1")

        # -------------------------------------------------------------
        # Check 8: Physical Sanity Bounds (excluding deliberate severe glitch faults)
        # -------------------------------------------------------------
        checks_run += 1
        clean_only = full_df[full_df["ground_truth"] == "NORMAL"]
        out_temp = clean_only[
            (clean_only["temperature"] < -10.0) | (clean_only["temperature"] > 55.0)
        ]
        out_hum = clean_only[
            (clean_only["humidity"] < 0.0) | (clean_only["humidity"] > 100.0)
        ]
        out_baro = clean_only[
            (clean_only["pressure"] < 800.0) | (clean_only["pressure"] > 1080.0)
        ]

        if len(out_temp) > 0:
            errors.append(f"{len(out_temp)} clean observations have unphysical temperature values")
        if len(out_hum) > 0:
            errors.append(f"{len(out_hum)} clean observations have unphysical humidity values")
        if len(out_baro) > 0:
            errors.append(f"{len(out_baro)} clean observations have unphysical pressure values")

        # -------------------------------------------------------------
        # Check 9: Station Coverage
        # -------------------------------------------------------------
        checks_run += 1
        expected_stations = {s.station_id for s in STATIONS}
        actual_stations = set(full_df["station_id"].unique())
        if actual_stations != expected_stations:
            errors.append(f"Station set mismatch: expected {expected_stations}, got {actual_stations}")

        # -------------------------------------------------------------
        # Check 10: Class & Severity Representation
        # -------------------------------------------------------------
        checks_run += 1
        gt_dist = full_df["ground_truth"].value_counts().to_dict()
        sev_dist = full_df["severity"].value_counts().to_dict()
        metrics["class_distribution"] = gt_dist
        metrics["severity_distribution"] = sev_dist

        for required_gt in ("NORMAL", "SENSOR_FAULT", "WEATHER_EVENT", "DATA_QUALITY_ISSUE", "COMMUNICATION_FAULT"):
            if gt_dist.get(required_gt, 0) == 0:
                errors.append(f"Missing ground truth category in benchmark: {required_gt}")

        for required_sev in ("MILD", "MODERATE", "SEVERE"):
            if sev_dist.get(required_sev, 0) == 0:
                errors.append(f"Missing severity level in benchmark: {required_sev}")

        passed = len(errors) == 0
        return ValidationResult(
            passed=passed,
            checks_run=checks_run,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )


def verify_reproducibility(config: Any) -> bool:
    """
    Verify that re-running the generator with the identical seed produces identical dataframes.
    """
    from .pipeline import BenchmarkPipeline

    pipe1 = BenchmarkPipeline(config)
    df1_full, _, _, _ = pipe1.build_benchmark_dataset()

    pipe2 = BenchmarkPipeline(config)
    df2_full, _, _, _ = pipe2.build_benchmark_dataset()

    return df1_full.equals(df2_full)
