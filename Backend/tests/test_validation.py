import pandas as pd

from skyguard_ml.validation import validate_observations


def test_invalid_timestamp_is_flagged():
    data = [
        {
            "station_id": "S1",
            "timestamp": "not-a-date",
            "temperature": 30,
        }
    ]

    frame, report = validate_observations(data)

    assert report.invalid_row_count == 1
    assert "INVALID_TIMESTAMP" in frame.iloc[0]["quality_flags"]


def test_duplicate_timestamps_are_flagged():
    data = [
        {
            "station_id": "S1",
            "timestamp": "2026-01-01T00:00:00Z",
            "temperature": 30,
        },
        {
            "station_id": "S1",
            "timestamp": "2026-01-01T00:00:00Z",
            "temperature": 31,
        },
    ]

    frame, report = validate_observations(data)

    assert report.invalid_row_count == 2
    assert all(
        "DUPLICATE_TIMESTAMP" in flags
        for flags in frame["quality_flags"]
    )
