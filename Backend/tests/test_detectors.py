import pandas as pd

from skyguard_ml.statistical_detectors import (
    detect_frozen_sensor,
    detect_spike_drop,
)


def test_temperature_spike_is_detected():
    row = pd.Series(
        {
            "temperature_delta": 8.0,
            "humidity_delta": 0.0,
            "pressure_delta": 0.0,
        }
    )

    result = detect_spike_drop(
        row,
        ["temperature", "humidity", "pressure"],
    )

    assert result["temperature"]["spike_score"] > 0.5


def test_frozen_sensor_is_detected():
    timestamps = pd.date_range(
        "2026-01-01",
        periods=10,
        freq="30min",
        tz="UTC",
    )

    history = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature": [30.0] * 10,
        }
    )

    result = detect_frozen_sensor(
        history,
        ["temperature"],
        duration_hours=2.0,
    )

    assert result["temperature"]["score"] > 0.0
