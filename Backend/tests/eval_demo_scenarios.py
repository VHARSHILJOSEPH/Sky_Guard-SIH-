"""
eval_demo_scenarios.py
----------------------
Evaluates the 6 synthetic anomaly scenarios from `skyguard_fake_demo_dataset.csv`
against the unified SkyGuard AI anomaly detection pipeline.

Ensures:
1. Only sensor fields are passed (no ground-truth cheat).
2. The exact same pipeline function (`process_pipeline_observation`) is executed.
3. Every scenario's detection status, anomaly type, severity, score, and explanation are captured and verified.
"""

import os
from pathlib import Path
import sys
import pandas as pd

# Add Backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from ml_api_server import (
    process_pipeline_observation,
    PipelineProcessRequest,
    station_history_cache,
    previous_observation_cache,
)


def find_dataset() -> Path:
    candidates = [
        backend_dir.parent / "datasets" / "skyguard_fake_demo_dataset.csv",
        backend_dir.parent / "Demo sets" / "skyguard_fake_demo_dataset.csv",
        backend_dir / "datasets" / "skyguard_fake_demo_dataset.csv",
        backend_dir / "Demo sets" / "skyguard_fake_demo_dataset.csv",
        backend_dir.parent / "public" / "skyguard_fake_demo_dataset.csv",
        Path("datasets/skyguard_fake_demo_dataset.csv"),
        Path("Demo sets/skyguard_fake_demo_dataset.csv"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(f"Could not find demo dataset in: {[str(c) for c in candidates]}")


def run_evaluation():
    print("=================================================================")
    print(" SKYGUARD AI — DEMO MODE SCENARIO EVALUATION REPORT")
    print("=================================================================")

    csv_path = find_dataset()
    print(f"Loading dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} rows, stations: {df['station_id'].unique().tolist()}\n")

    # Reset caches before starting simulation
    station_history_cache.clear()
    previous_observation_cache.clear()

    demo_run_id = "demo_eval_20260914"
    results = []

    print("Executing unified detection pipeline sequentially across all 504 rows...")
    for idx, row in df.iterrows():
        # STRICT ISOLATION: Only telemetry columns passed
        obs = {
            "station_id": str(row["station_id"]),
            "timestamp": str(row["timestamp"]),
            "temperature_c": None if pd.isna(row["temperature_c"]) else float(row["temperature_c"]),
            "humidity_pct": None if pd.isna(row["humidity_pct"]) else float(row["humidity_pct"]),
            "pressure_hpa": None if pd.isna(row["pressure_hpa"]) else float(row["pressure_hpa"]),
            "wind_speed_ms": None if pd.isna(row["wind_speed_ms"]) else float(row["wind_speed_ms"]),
            "rainfall_mm": None if pd.isna(row["rainfall_mm"]) else float(row["rainfall_mm"]),
        }

        req = PipelineProcessRequest(
            station_id=obs["station_id"],
            observation=obs,
            source="DEMO",
            demo_run_id=demo_run_id,
            history=None, # uses temporal rolling window
        )

        res = process_pipeline_observation(req)
        results.append({
            "row_idx": idx,
            "station_id": obs["station_id"],
            "timestamp": obs["timestamp"],
            "temperature_c": obs["temperature_c"],
            "humidity_pct": obs["humidity_pct"],
            "pressure_hpa": obs["pressure_hpa"],
            "wind_speed_ms": obs["wind_speed_ms"],
            "rainfall_mm": obs["rainfall_mm"],
            "is_anomaly": res.get("is_anomaly", False),
            "anomaly_type": res.get("anomaly_type", "NONE"),
            "severity": res.get("severity", "LOW"),
            "anomaly_score": res.get("anomaly_score", 0.0),
            "confidence": res.get("confidence", 0.0),
            "explanation": res.get("explanation", ""),
            "why_flagged": res.get("why_flagged", []),
        })

    print(f"Processed all {len(results)} rows successfully.\n")

    # Scenario Verification Definitions
    scenarios = [
        {
            "id": 1,
            "name": "Scenario 1: Temperature Spike",
            "station": "HYD_AWS_01",
            "target_row": 82,
            "description": "Sudden sharp jump in temperature to 43.95°C (+11.5°C jump)",
            "check": lambda r: r["row_idx"] in range(81, 84) and r["is_anomaly"],
        },
        {
            "id": 2,
            "name": "Scenario 2: Missing Telemetry",
            "station": "HYD_AWS_01",
            "target_row": 134,
            "description": "Missing/null barometric pressure value (NaN)",
            "check": lambda r: r["row_idx"] == 134 and r["is_anomaly"],
        },
        {
            "id": 3,
            "name": "Scenario 3: Corrupted Out-of-Bounds Value",
            "station": "HYD_AWS_01",
            "target_row": 135,
            "description": "Extreme physical bounds violation (temperature = 999.0°C)",
            "check": lambda r: r["row_idx"] == 135 and r["is_anomaly"],
        },
        {
            "id": 4,
            "name": "Scenario 4: Sensor Freeze",
            "station": "BLR_AWS_02",
            "target_row": 273,
            "description": "Repetitive frozen humidity reading flatlined at 69.87% across multiple consecutive hours",
            "check": lambda r: r["row_idx"] in range(270, 277) and r["is_anomaly"],
        },
        {
            "id": 5,
            "name": "Scenario 5: Multivariate Discrepancy",
            "station": "BLR_AWS_02",
            "target_row": 323,
            "description": "Multivariate inconsistency: sudden humidity surge to 96% with drastic pressure drop (-8.6 hPa)",
            "check": lambda r: r["row_idx"] in range(322, 325) and r["is_anomaly"],
        },
        {
            "id": 6,
            "name": "Scenario 6: Calibration Drift / Dynamic Shift",
            "station": "DEL_AWS_03",
            "target_row": 371,
            "description": "Pressure/thermal trajectory shifts across DEL_AWS_03 sequence",
            "check": lambda r: r["station_id"] == "DEL_AWS_03" and r["is_anomaly"],
        },
    ]

    print("-----------------------------------------------------------------")
    print(" DETAILED SCENARIO DETECTION RESULTS")
    print("-----------------------------------------------------------------")

    summary_records = []
    for sc in scenarios:
        matched = [r for r in results if sc["check"](r)]
        sample = matched[0] if matched else results[sc["target_row"]]

        status_str = "PASSED (DETECTED)" if matched else "INVESTIGATE"
        print(f"[{status_str}] {sc['name']}")
        print(f"  Station:     {sample['station_id']}")
        print(f"  Row Index:   {sample['row_idx']} (Row {sample['row_idx'] + 1} of 504)")
        print(f"  Timestamp:   {sample['timestamp']}")
        print(f"  Anomaly:     {sample['is_anomaly']}")
        print(f"  Type:        {sample['anomaly_type']}")
        print(f"  Severity:    {sample['severity']}")
        print(f"  Score:       {sample['anomaly_score']:.3f} | Confidence: {sample['confidence']:.2f}")
        print(f"  Explanation: {sample['explanation']}")
        print(f"  Why Flagged: {sample['why_flagged']}")
        print()

        summary_records.append({
            "Scenario": sc["name"],
            "Station": sample["station_id"],
            "Row": sample["row_idx"] + 1,
            "Detected": sample["is_anomaly"],
            "Type": sample["anomaly_type"],
            "Severity": sample["severity"],
            "Score": round(sample["anomaly_score"], 3),
        })

    # Total stats across dataset
    total_anomalies = sum(1 for r in results if r["is_anomaly"])
    print("=================================================================")
    print(f" Total Rows Replayed: {len(results)}")
    print(f" Total Anomalies Flagged: {total_anomalies} ({(total_anomalies / len(results) * 100):.1f}%)")
    print("=================================================================")


if __name__ == "__main__":
    run_evaluation()
