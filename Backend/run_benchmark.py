"""
SkyGuard AI — Master Benchmark Runner CLI.

Generates the benchmark datasets, runs quality checks and non-leakage validations,
executes candidate model evaluations (LOF, Isolation Forest, One-Class SVM),
and exports the benchmark reports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure Backend is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from skyguard_ml.benchmark.config import BenchmarkConfig
from skyguard_ml.benchmark.pipeline import BenchmarkPipeline
from skyguard_ml.benchmark.validator import BenchmarkValidator, verify_reproducibility
from skyguard_ml.benchmark.evaluator import BenchmarkEvaluator
from skyguard_ml.benchmark.reporting import generate_benchmark_reports


def run():
    print("=" * 70)
    print("SkyGuard AI: Scientific Anomaly Detection Benchmark Pipeline")
    print("=" * 70)

    config = BenchmarkConfig(random_seed=42)
    pipeline = BenchmarkPipeline(config)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 1. Generate Datasets
    print("\n[Step 1/5] Generating benchmark datasets...")
    full_df, train_df, val_df, test_df = pipeline.build_benchmark_dataset()
    print(f"  -> Full dataset: {len(full_df)} observations across {full_df['station_id'].nunique()} stations")
    print(f"  -> Train split: {len(train_df)} observations ({len(train_df)/len(full_df)*100:.1f}%) [Clean Normal Only]")
    print(f"  -> Validation split: {len(val_df)} observations ({len(val_df)/len(full_df)*100:.1f}%) [Threshold Calibration]")
    print(f"  -> Test split: {len(test_df)} observations ({len(test_df)/len(full_df)*100:.1f}%) [Frozen Benchmark]")

    # 2. Automated Validation
    print("\n[Step 2/5] Running scientific dataset validation suite...")
    validator = BenchmarkValidator(config.benchmark_dir)
    val_result = validator.run_all_checks(full_df, train_df, val_df, test_df)

    if not val_result.passed:
        print("  [FAIL] VALIDATION FAILED with errors:")
        for err in val_result.errors:
            print(f"     - {err}")
        sys.exit(1)
    else:
        print(f"  [PASS] All {val_result.checks_run} validation checks PASSED successfully.")
        print("     - Zero train contamination: 100% clean normal observations")
        print("     - Zero scenario leakage between train, validation, and test")
        print("     - Strictly chronological split boundaries confirmed")

    # 3. Verify Reproducibility
    print("\n[Step 3/5] Verifying deterministic reproducibility with seed=42...")
    is_reproducible = verify_reproducibility(config)
    if is_reproducible:
        print("  [PASS] Deterministic reproducibility CONFIRMED: identical byte-for-byte generation.")
    else:
        print("  [FAIL] Deterministic reproducibility FAILED.")
        sys.exit(1)

    # 4. Model Benchmarking
    print("\n[Step 4/5] Benchmarking unsupervised detectors on benchmark splits...")
    print("     Models: Local Outlier Factor (LOF), Isolation Forest, One-Class SVM")
    evaluator = BenchmarkEvaluator(random_state=config.random_seed)
    eval_results = evaluator.run_benchmark_suite(train_df, val_df, test_df)

    for model_name, res in eval_results.items():
        print(f"\n  --- {model_name} Performance ---")
        print(f"      Calibrated Threshold: {res['calibrated_threshold']:.4f} (Validation F1: {res['validation_f1']:.4f})")
        print(f"      Test Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f} | F1: {res['f1']:.4f}")
        print(f"      Specificity: {res['specificity']:.4f} | Balanced Accuracy: {res['balanced_accuracy']:.4f}")
        print(f"      ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f} | MCC: {res['mcc']:.4f}")
        if res.get("false_alarms_per_station_day") is not None:
            print(f"      False Alarms/Station/Day: {res['false_alarms_per_station_day']:.2f}")
        if res.get("mean_detection_latency_seconds") is not None:
            print(f"      Mean Detection Latency: {res['mean_detection_latency_seconds']:.1f} seconds")

    # 5. Export Reports
    print("\n[Step 5/5] Generating benchmark reports...")
    metadata_path = config.benchmark_dir / "benchmark_metadata.json"
    import json
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    json_path, md_path = generate_benchmark_reports(metadata, eval_results, config.benchmark_dir)
    print(f"  -> Machine-readable report saved to: {json_path}")
    print(f"  -> Human-readable report saved to: {md_path}")

    # Representative examples
    print("\n" + "=" * 70)
    print("Representative Examples from Benchmark Scenarios")
    print("=" * 70)

    sample_scenarios = [
        ("CLEAN_NORMAL", "SCN_CLEAN_BASELINE"),
        ("SINGLE_STATION_FAULT", "SCN_TEST_CASE_A_ISOLATED_DRIFT_43189"),
        ("MULTI_STATION_WEATHER_EVENT", "SCN_TEST_CASE_B_COHERENT_STORM_DELTA"),
        ("DATA_QUALITY", "SCN_TEST_DUPLICATE_43189"),
        ("COMMUNICATION", "SCN_TEST_COMM_FAIL_43245"),
    ]

    for scn_type, scn_id in sample_scenarios:
        match = full_df[full_df["scenario_id"] == scn_id]
        if not match.empty:
            row = match.iloc[len(match) // 2]
            print(f"\nScenario Type: {scn_type} | ID: {scn_id}")
            print(f"  Station: {row['station_id']} ({row['station_name']}) at {row['timestamp']}")
            print(f"  Telemetry: Temp={row['temperature']}C, Humidity={row['humidity']}%, Baro={row['pressure']} hPa")
            print(f"  Ground Truth: {row['ground_truth']} | Fault Type: {row['fault_type']} | Severity: {row['severity']}")
            print(f"  Is Anomaly: {row['is_anomaly']} | Is Sensor Fault: {row['is_sensor_fault']}")
            print(f"  Description: {row['description']}")

    print("\n" + "=" * 70)
    print("Benchmark generation, validation, and evaluation successfully completed!")
    print("=" * 70)


if __name__ == "__main__":
    run()
