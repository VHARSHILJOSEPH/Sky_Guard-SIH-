"""
SkyGuard AI — Phase 8 Production Training & Model Comparison Runner

Executes the audited ML training workflow:
1. Generates clean training baseline and multi-station observations (SIH core variables: Temperature, Humidity, Pressure).
2. Performs chronological split (60% train, 20% validation, 20% test).
3. Injects multi-severity synthetic faults into validation and test sets.
4. Compares all 4 candidate models (Isolation Forest, LOF, One-Class SVM, Robust Covariance).
5. Calibrates operating threshold for minimal false positive rate (< 0.05).
6. Persists calibrated production artifacts (v2.0.0).
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from skyguard_ml.config import TrainingConfig
from skyguard_ml.model_comparison import run_model_comparison
from skyguard_ml.training import train_all
from train_lof_model import generate_clean_station_telemetry


def generate_production_training_dataset() -> list[dict]:
    """Generate extensive multi-station Indian AWS observation dataset."""
    stations = [
        {"id": "43189", "name": "Vijayawada (AWS014)", "temp": 32.4, "hum": 64.0, "baro": 1008.2},
        {"id": "43150", "name": "Visakhapatnam (AWS008)", "temp": 29.8, "hum": 78.0, "baro": 1012.4},
        {"id": "43245", "name": "Tirupati (AWS021)", "temp": 34.1, "hum": 52.0, "baro": 1004.8},
        {"id": "43110", "name": "Hyderabad (AWS001)", "temp": 31.0, "hum": 58.0, "baro": 1009.5},
    ]

    all_records = []
    # 240 hours (10 days) per station = 960 clean observations
    for stn in stations:
        recs = generate_clean_station_telemetry(
            station_id=stn["id"],
            base_temp=stn["temp"],
            base_hum=stn["hum"],
            base_baro=stn["baro"],
            n_hours=240,
            start_date="2026-08-01T00:00:00Z",
        )
        all_records.extend(recs)

    return all_records


def main():
    print("=" * 70)
    print("SKYGUARD AI — PHASE 8: PRODUCTION TRAINING & MODEL COMPARISON")
    print("=" * 70)

    dataset = generate_production_training_dataset()
    print(f"Loaded {len(dataset)} clean hourly observations across 4 AWS stations.")

    artifact_dir = Path("skyguard_ml/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = TrainingConfig(
        synthetic_faults_per_type=4,
        min_training_rows=50,
        artifact_dir=str(artifact_dir),
        random_state=42,
    )

    # 1. Run comparison across all 4 models
    print("\n[Step 1/3] Running Fair Model Comparison across 4 Candidates...")
    comparison = run_model_comparison(
        observations=dataset,
        config=config,
        output_dir=artifact_dir,
    )

    print("\nCandidate Model Comparison Results:")
    print(comparison["markdown_table"])
    print(f"\nWinning Model Selected: {comparison['winning_model']}")

    # 2. Train and serialize production bundle
    print("\n[Step 2/3] Training and Serializing Production Model Artifacts (v2.0.0)...")
    train_report = train_all(
        observations=dataset,
        config=config,
        artifact_dir=artifact_dir,
    )

    print(f"Trained and saved to: {artifact_dir.resolve()}")
    print(f"Selected Model: {train_report['selected_model']}")
    print(f"Feature Count: {train_report['feature_count']} (SIH core variables only)")

    # 3. Verify Artifacts
    print("\n[Step 3/3] Verifying Production Artifacts...")
    expected_files = [
        "best_model.joblib",
        "candidate_models.joblib",
        "feature_config.json",
        "preprocessing_config.json",
        "baseline_statistics.json",
        "fusion_weights.json",
        "model_metadata.json",
        "evaluation_report.json",
        "model_comparison_report.json",
        "model_comparison_report.md",
    ]

    for fname in expected_files:
        fpath = artifact_dir / fname
        assert fpath.exists(), f"Missing artifact: {fname}"
        print(f"  [OK] {fname} ({fpath.stat().st_size} bytes)")

    meta = json.loads((artifact_dir / "model_metadata.json").read_text())
    print("\nFinal Calibrated Operational Metrics (Untouched Test Set):")
    for k, v in meta["test_metrics"].items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.4f}")
        elif v is not None and not isinstance(v, dict):
            print(f"  - {k}: {v}")

    print("\n" + "=" * 70)
    print("PHASE 8 TRAINING & SELECTION COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
