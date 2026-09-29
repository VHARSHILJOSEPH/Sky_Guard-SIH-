"""
SkyGuard AI — LOF Training & Threshold Calibration Script

Trains the production Local Outlier Factor (LOF) anomaly detector for ESP32 + BME280 telemetry.
Strictly adheres to SkyGuard ML rules:
1. Normal, clean observations ONLY for model fitting (no synthetic faults in training).
2. Synthetic faults injected ONLY for validation-based threshold calibration & test benchmarking.
3. Causal historical feature engineering (no future-data leakage).
4. Persists the frozen, calibrated model artifact using joblib.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import numpy as np
import pandas as pd

from skyguard_lof import (
    SkyGuardLOF,
    SkyGuardLOFConfig,
    build_feature_dataset,
    inject_synthetic_fault,
    evaluate_synthetic_records,
)

ARTIFACT_PATH = Path(os.environ.get("SKYGUARD_MODEL_PATH", "skyguard_lof_bme280_calibrated.joblib"))


def generate_clean_station_telemetry(
    station_id: str,
    base_temp: float,
    base_hum: float,
    base_baro: float,
    n_hours: int = 120,
    start_date: str = "2026-08-01T00:00:00Z",
) -> list[dict]:
    """Generate realistic diurnal hourly telemetry for a clean weather station."""
    records = []
    base_time = pd.Timestamp(start_date)
    rng = np.random.default_rng(seed=abs(hash(station_id)) % (2**31))

    for h in range(n_hours):
        t = base_time + pd.Timedelta(hours=h)
        hour_of_day = t.hour

        # Diurnal temperature cycle: peak around 14:00, trough around 05:00
        diurnal_temp = 5.0 * math.sin((hour_of_day - 8) * math.pi / 12)
        temp = base_temp + diurnal_temp + rng.normal(0, 0.4)

        # Diurnal humidity cycle: inverse to temperature
        diurnal_hum = -12.0 * math.sin((hour_of_day - 8) * math.pi / 12)
        hum = float(np.clip(base_hum + diurnal_hum + rng.normal(0, 1.2), 15.0, 98.0))

        # Diurnal atmospheric tide: semi-diurnal barometric variation (~1.5 hPa)
        baro_cycle = 1.2 * math.cos(hour_of_day * math.pi / 6)
        baro = base_baro + baro_cycle + rng.normal(0, 0.25)

        records.append({
            "station_id": station_id,
            "timestamp": t.isoformat(),
            "temperature": round(temp, 2),
            "humidity": round(hum, 1),
            "pressure": round(baro, 2),
            "is_anomaly": 0,
        })

    return records


def main():
    print("=" * 60)
    print("SkyGuard AI: Training & Calibrating LOF Model")
    print("=" * 60)

    # 1. Generate clean training observations across standard Indian AWS stations
    stations = [
        {"id": "43189", "name": "Vijayawada (AWS014)", "temp": 32.4, "hum": 64.0, "baro": 1008.2},
        {"id": "43150", "name": "Visakhapatnam (AWS008)", "temp": 29.8, "hum": 78.0, "baro": 1012.4},
        {"id": "43245", "name": "Tirupati (AWS021)", "temp": 34.1, "hum": 52.0, "baro": 1004.8},
    ]

    all_clean_train = []
    all_validation_records = []
    all_test_records = []

    for stn in stations:
        # 120 hours of clean observations for training (5 days)
        train_recs = generate_clean_station_telemetry(
            station_id=stn["id"],
            base_temp=stn["temp"],
            base_hum=stn["hum"],
            base_baro=stn["baro"],
            n_hours=120,
            start_date="2026-08-01T00:00:00Z",
        )
        all_clean_train.extend(train_recs)

        # 80 hours for validation (threshold calibration with synthetic faults)
        val_clean = generate_clean_station_telemetry(
            station_id=stn["id"],
            base_temp=stn["temp"],
            base_hum=stn["hum"],
            base_baro=stn["baro"],
            n_hours=80,
            start_date="2026-08-10T00:00:00Z",
        )
        # Inject validation synthetic faults (e.g. spike, drop, drift, frozen)
        val_faulted = inject_synthetic_fault(
            val_clean,
            fault_type="TEMPERATURE_SPIKE",
            start_index=35,
            duration=4,
            magnitude=8.5,
            fault_id=f"val_spike_{stn['id']}",
        )
        val_faulted = inject_synthetic_fault(
            val_faulted,
            fault_type="FROZEN_SENSOR",
            start_index=55,
            duration=6,
            fault_id=f"val_frozen_{stn['id']}",
        )
        all_validation_records.extend(val_faulted)

        # 80 hours for test benchmarking
        test_clean = generate_clean_station_telemetry(
            station_id=stn["id"],
            base_temp=stn["temp"],
            base_hum=stn["hum"],
            base_baro=stn["baro"],
            n_hours=80,
            start_date="2026-08-20T00:00:00Z",
        )
        test_faulted = inject_synthetic_fault(
            test_clean,
            fault_type="HUMIDITY_ANOMALY",
            start_index=30,
            duration=5,
            magnitude=28.0,
            fault_id=f"test_hum_{stn['id']}",
        )
        test_faulted = inject_synthetic_fault(
            test_faulted,
            fault_type="PRESSURE_ANOMALY",
            start_index=50,
            duration=4,
            magnitude=14.0,
            fault_id=f"test_baro_{stn['id']}",
        )
        all_test_records.extend(test_faulted)

    # 2. Configure model
    config = SkyGuardLOFConfig(
        n_neighbors=20,
        contamination="auto",
        rolling_window_hours=24,
        min_history_length=24,
        threshold_method="validation_f1",
        random_state=42,
    )
    model = SkyGuardLOF(config)

    # 3. Fit on CLEAN training data ONLY
    print(f"Fitting SkyGuardLOF on {len(all_clean_train)} clean training observations...")
    model.fit(all_clean_train, dataset_version="bme280_clean_train_v1")
    print(f"Fitted in {model.training_time_seconds:.3f}s. Feature rows: {model.training_feature_row_count}")

    # 4. Calibrate threshold using validation dataset with synthetic faults
    print("Building validation dataset for threshold calibration...")
    val_dataset = build_feature_dataset(
        all_validation_records,
        labels=[r.get("is_anomaly", 0) for r in all_validation_records],
        config=config,
    )
    print(f"Validation dataset rows: {len(val_dataset.X)}, anomalies: {int(np.sum(val_dataset.y))}")

    calibration_metrics = model.calibrate_threshold(val_dataset.X, val_dataset.y)
    print(f"Calibrated Threshold: {model.threshold:.4f}")
    print(f"Validation F1: {calibration_metrics.get('f1', 0):.4f}, Precision: {calibration_metrics.get('precision', 0):.4f}, Recall: {calibration_metrics.get('recall', 0):.4f}")

    # 5. Evaluate on test set (benchmark-only metrics)
    print("Evaluating on independent test dataset...")
    test_report = evaluate_synthetic_records(model, all_test_records)
    print(f"Test Accuracy: {test_report.get('accuracy', 0):.4f}, F1: {test_report.get('f1', 0):.4f}")
    print(f"Test Precision: {test_report.get('precision', 0):.4f}, Recall: {test_report.get('recall', 0):.4f}")
    if test_report.get("false_alarms_per_day") is not None:
        print(f"False Alarms/Day: {test_report.get('false_alarms_per_day', 0):.2f}")

    # 6. Save model artifact
    output_path = ARTIFACT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"\nModel successfully saved to: {output_path.resolve()}")

    # 7. Verify save/load
    loaded = SkyGuardLOF.load(output_path)
    info = loaded.get_model_info()
    print("Verification: Loaded model version:", info["model_version"])
    print("Threshold:", info["threshold"])
    print("Features count:", len(info["feature_names"]))
    print("=" * 60)
    print("Training and calibration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
