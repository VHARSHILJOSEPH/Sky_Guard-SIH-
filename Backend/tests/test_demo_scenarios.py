import csv
import sys
from pathlib import Path

# Add Backend to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from ml_api_server import (
    process_pipeline_observation,
    PipelineProcessRequest,
    station_history_cache,
    previous_observation_cache,
)

def test_all_demo_scenarios():
    dataset_path = None
    for p in [
        backend_dir.parent / "datasets" / "skyguard_fake_demo_dataset.csv",
        backend_dir.parent / "Demo sets" / "skyguard_fake_demo_dataset.csv",
        backend_dir / "datasets" / "skyguard_fake_demo_dataset.csv",
    ]:
        if p.exists():
            dataset_path = p
            break
    assert dataset_path is not None, "Dataset not found in datasets/ or Demo sets/"

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            records.append((idx, row))

    # Targeted scenarios: name, station, target timestamp, window of preceding rows to feed
    scenarios = [
        ("SPIKE", "HYD_AWS_01", "2026-09-04 10:00:00+00:00", ["TEMPERATURE_SPIKE", "TEMPERATURE_ANOMALY", "SENSOR_DRIFT"], 30),
        ("FROZEN_SENSOR", "BLR_AWS_02", "2026-09-05 12:00:00+00:00", ["FROZEN_SENSOR", "LIKELY_SENSOR_ANOMALY"], 30),
        ("DRIFT", "DEL_AWS_03", "2026-09-04 17:00:00+00:00", ["TEMPERATURE_DRIFT", "DRIFT", "UNKNOWN_ANOMALY", "MULTIVARIATE_INCONSISTENCY", "SENSOR_DRIFT", "LIKELY_SENSOR_ANOMALY"], 30),
        ("MISSING_DATA", "HYD_AWS_01", "2026-09-06 14:00:00+00:00", ["MISSING_DATA", "DATA_QUALITY_ANOMALY"], 30),
        ("CORRUPTED_DATA", "HYD_AWS_01", "2026-09-06 15:00:00+00:00", ["CORRUPTED_DATA", "DATA_QUALITY_ANOMALY"], 30),
        ("MULTIVARIATE_INCONSISTENCY", "BLR_AWS_02", "2026-09-07 11:00:00+00:00", ["MULTIVARIATE_INCONSISTENCY", "SENSOR_DRIFT", "LIKELY_SENSOR_ANOMALY"], 30),
    ]

    results = {}

    for name, stn, target_ts, expected_types, window_len in scenarios:
        # Clear cache for isolated test
        station_history_cache.clear()
        previous_observation_cache.clear()

        stn_rows = [r for r in records if r[1]["station_id"] == stn]
        
        # Find index of target_ts
        target_idx = -1
        for idx, (_, r) in enumerate(stn_rows):
            if r["timestamp"] == target_ts:
                target_idx = idx
                break
        
        assert target_idx != -1, f"Target timestamp {target_ts} not found for station {stn}"

        # Feed preceding window_len observations + target observation
        start_idx = max(0, target_idx - window_len)
        test_slice = stn_rows[start_idx : target_idx + 1]

        target_res = None
        for _, r in test_slice:
            obs = {}
            for k, v in r.items():
                if k in ("station_id", "timestamp"):
                    continue
                if v in ("", "NaN", "null", None):
                    obs[k] = None
                else:
                    try:
                        obs[k] = float(v)
                    except ValueError:
                        obs[k] = v
            obs["timestamp"] = r["timestamp"]

            req = PipelineProcessRequest(station_id=stn, observation=obs, source="DEMO")
            res = process_pipeline_observation(req)

            if r["timestamp"] == target_ts:
                target_res = res

        assert target_res is not None, f"Failed to get result for target {target_ts}"
        
        results[name] = {
            "scenario": name,
            "target_ts": target_ts,
            "is_anomaly": target_res["is_anomaly"],
            "anomaly_type": target_res["anomaly_type"],
            "severity": target_res["severity"],
            "ai_status": target_res["ai_status"],
            "score": target_res["anomaly_score"],
            "change": target_res["change"],
            "expected_types": expected_types,
            "matched": target_res["anomaly_type"] in expected_types and target_res["is_anomaly"],
        }

    print("\n=== SCENARIO DETECTION RESULTS ===")
    all_passed = True
    for name, data in results.items():
        status_symbol = "PASS" if data["matched"] else "FAIL"
        if not data["matched"]:
            all_passed = False
        safe_ai_status = data['ai_status'].encode('ascii', 'ignore').decode('ascii').strip()
        print(f"[{status_symbol}] {name}: type={data['anomaly_type']}, is_anomaly={data['is_anomaly']}, severity={data['severity']}, ai_status={safe_ai_status}")
    
    assert all_passed, f"Some scenarios failed: {results}"
    print("\nALL 6 DEMO SCENARIOS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_all_demo_scenarios()
