"""
SkyGuard AI — Automated Tests for Scientific Benchmark Dataset Pipeline.

Verifies:
1. Complete directory structure: raw, processed, clean, synthetic_faults, weather_events, benchmark.
2. Zero training split contamination (100% clean normal).
3. Strictly chronological train/val/test splits without scenario leakage.
4. Paired Case A (single station fault) and Case B (multi-station weather event) scenarios.
5. Ground truth taxonomy and label consistency.
6. Deterministic reproducibility.
7. Candidate model evaluation support (LOF, Isolation Forest, One-Class SVM).
"""

import json
from pathlib import Path
import pytest
import pandas as pd

from skyguard_ml.benchmark.config import BenchmarkConfig, STATIONS
from skyguard_ml.benchmark.pipeline import BenchmarkPipeline
from skyguard_ml.benchmark.validator import BenchmarkValidator, verify_reproducibility
from skyguard_ml.benchmark.evaluator import BenchmarkEvaluator


@pytest.fixture(scope="module")
def benchmark_data():
    config = BenchmarkConfig(random_seed=42)
    pipeline = BenchmarkPipeline(config)
    full_df, train_df, val_df, test_df = pipeline.build_benchmark_dataset()
    return {
        "config": config,
        "full_df": full_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }


def test_directory_structure(benchmark_data):
    config = benchmark_data["config"]
    assert config.raw_dir.exists(), "datasets/raw directory missing"
    assert config.processed_dir.exists(), "datasets/processed directory missing"
    assert config.clean_dir.exists(), "datasets/clean directory missing"
    assert config.synthetic_faults_dir.exists(), "datasets/synthetic_faults directory missing"
    assert config.weather_events_dir.exists(), "datasets/weather_events directory missing"
    assert config.benchmark_dir.exists(), "datasets/benchmark directory missing"

    # Core artifacts exist
    assert (config.clean_dir / "clean_telemetry.csv").exists()
    assert (config.synthetic_faults_dir / "synthetic_faults.csv").exists()
    assert (config.weather_events_dir / "weather_events.csv").exists()
    assert (config.benchmark_dir / "benchmark_dataset.csv").exists()
    assert (config.benchmark_dir / "benchmark_train.csv").exists()
    assert (config.benchmark_dir / "benchmark_val.csv").exists()
    assert (config.benchmark_dir / "benchmark_test.csv").exists()
    assert (config.benchmark_dir / "benchmark_metadata.json").exists()


def test_zero_train_contamination(benchmark_data):
    train_df = benchmark_data["train_df"]
    assert (train_df["ground_truth"] == "NORMAL").all(), "Train split contains non-normal ground truth"
    assert (train_df["is_anomaly"] == 0).all(), "Train split contains anomaly flags"
    assert (train_df["is_sensor_fault"] == 0).all(), "Train split contains sensor fault flags"
    assert (train_df["fault_type"] == "NONE").all(), "Train split contains injected fault types"


def test_chronological_splits_and_no_leakage(benchmark_data):
    train_df = benchmark_data["train_df"]
    val_df = benchmark_data["val_df"]
    test_df = benchmark_data["test_df"]

    train_max = pd.to_datetime(train_df["timestamp"], utc=True).max()
    val_min = pd.to_datetime(val_df["timestamp"], utc=True).min()
    val_max = pd.to_datetime(val_df["timestamp"], utc=True).max()
    test_min = pd.to_datetime(test_df["timestamp"], utc=True).min()

    assert train_max < val_min, f"Train ({train_max}) overlaps validation ({val_min})"
    assert val_max < test_min, f"Validation ({val_max}) overlaps test ({test_min})"

    # Scenario IDs must not leak
    val_scns = set(val_df["scenario_id"].unique()) - {"SCN_CLEAN_BASELINE"}
    test_scns = set(test_df["scenario_id"].unique()) - {"SCN_CLEAN_BASELINE"}
    assert len(val_scns.intersection(test_scns)) == 0, "Scenario leakage detected between val and test"


def test_case_a_and_case_b_scenarios(benchmark_data):
    test_df = benchmark_data["test_df"]

    # CASE A: Isolated station fault
    case_a_rows = test_df[test_df["scenario_id"] == "SCN_TEST_CASE_A_ISOLATED_DRIFT_43189"]
    assert not case_a_rows.empty, "Case A scenario missing"
    assert (case_a_rows["station_id"] == "43189").all()
    assert (case_a_rows["ground_truth"] == "SENSOR_FAULT").all()

    # CASE B: Coherent regional storm across 3 delta stations
    case_b_rows = test_df[test_df["scenario_id"] == "SCN_TEST_CASE_B_COHERENT_STORM_DELTA"]
    assert not case_b_rows.empty, "Case B scenario missing"
    assert set(case_b_rows["station_id"].unique()) == {"43189", "AWS002", "AWS004"}
    assert (case_b_rows["ground_truth"] == "WEATHER_EVENT").all()
    assert (case_b_rows["is_sensor_fault"] == 0).all()  # Must not be labeled as sensor fault!


def test_validator_passes(benchmark_data):
    config = benchmark_data["config"]
    validator = BenchmarkValidator(config.benchmark_dir)
    res = validator.run_all_checks(
        benchmark_data["full_df"],
        benchmark_data["train_df"],
        benchmark_data["val_df"],
        benchmark_data["test_df"],
    )
    assert res.passed, f"Validation failed with errors: {res.errors}"
    assert res.checks_run >= 10


def test_reproducibility(benchmark_data):
    config = benchmark_data["config"]
    assert verify_reproducibility(config), "Benchmark generation is not reproducible with fixed seed"


def test_evaluator_metrics(benchmark_data):
    evaluator = BenchmarkEvaluator(random_state=42)
    res = evaluator.evaluate_detector(
        "LOF",
        benchmark_data["train_df"],
        benchmark_data["val_df"],
        benchmark_data["test_df"],
    )
    # Verify all 12 metrics are present and valid
    for metric_name in (
        "precision",
        "recall",
        "f1",
        "specificity",
        "fpr",
        "fnr",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "pr_auc",
        "false_alarms_per_station_day",
        "calibrated_threshold",
    ):
        assert metric_name in res, f"Metric {metric_name} missing from evaluation results"
        assert res[metric_name] is not None

    assert res["f1"] > 0.40, f"LOF F1 score ({res['f1']}) unexpectedly low"
    assert res["specificity"] > 0.70, f"LOF Specificity ({res['specificity']}) unexpectedly low"
