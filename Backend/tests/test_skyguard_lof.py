import copy
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure Backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# pyrefly: ignore [missing-import]
from skyguard_lof import (
    SkyGuardLOF,
    SkyGuardLOFConfig,
    build_features,
    build_feature_dataset,
)


def make_observations(count=30):
    records = []
    base = pd.Timestamp("2026-09-01T00:00:00Z")

    for index in range(count):
        records.append(
            {
                "station_id": "ESP32_01",
                "timestamp": (base + pd.Timedelta(hours=index)).isoformat(),
                "temperature": 30.0 + index * 0.1,
                "humidity": 60.0 + index * 0.2,
                "pressure": 1008.0 + index * 0.05,
            }
        )

    return records


def test_future_observations_do_not_change_current_features():
    records = make_observations(35)

    config = SkyGuardLOFConfig(
        min_history_length=24,
        rolling_window_hours=24,
    )

    current = records[24]
    history = records[:24]

    original_features = build_features(
        history,
        current,
        station_id="ESP32_01",
        config=config,
    )

    records_with_future_changes = copy.deepcopy(records)
    records_with_future_changes[25]["temperature"] = 999.0
    records_with_future_changes[26]["humidity"] = 0.0
    records_with_future_changes[27]["pressure"] = 2000.0

    features_after_future_changes = build_features(
        records_with_future_changes,
        current,
        station_id="ESP32_01",
        config=config,
    )

    assert original_features == features_after_future_changes


def test_feature_count():
    records = make_observations(30)

    dataset = build_feature_dataset(
        records,
        config=SkyGuardLOFConfig(min_history_length=24),
    )

    assert dataset.X.shape[1] == 21
    assert len(dataset.X.columns) == 21


def test_station_history_is_not_mixed():
    records = make_observations(40)

    model = SkyGuardLOF(
        SkyGuardLOFConfig(min_history_length=24, n_neighbors=10)
    )
    dataset = build_feature_dataset(
        records[:35],
        config=model.config,
    )
    model.fit(dataset.X)

    test_history = copy.deepcopy(records[:29])
    test_history[0]["station_id"] = "ESP32_02"

    result = model.predict(
        station_id="ESP32_01",
        observation=records[29],
        history=test_history,
    )

    assert result["status"] == "INVALID_INPUT"


def test_model_save_load():
    records = make_observations(80)

    model = SkyGuardLOF(
        SkyGuardLOFConfig(
            min_history_length=24,
            n_neighbors=10,
        )
    )

    dataset = build_feature_dataset(
        records,
        config=model.config,
    )

    model.fit(dataset.X)

    labels = np.zeros(len(dataset.X), dtype=int)
    labels[-5:] = 1

    model.calibrate_threshold(dataset.X, labels)

    result_before = model.predict(
        station_id="ESP32_01",
        observation=records[-1],
        history=records[:-1],
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        model.save(path)

        loaded = SkyGuardLOF.load(path)

        result_after = loaded.predict(
            station_id="ESP32_01",
            observation=records[-1],
            history=records[:-1],
        )

    assert result_before["raw_lof_score"] == result_after["raw_lof_score"]
    assert (
        result_before["normalized_lof_score"]
        == result_after["normalized_lof_score"]
    )
    assert result_before["ml_anomaly"] == result_after["ml_anomaly"]


if __name__ == "__main__":
    test_future_observations_do_not_change_current_features()
    test_feature_count()
    test_station_history_is_not_mixed()
    test_model_save_load()
    print("All SkyGuard LOF tests passed successfully!")
