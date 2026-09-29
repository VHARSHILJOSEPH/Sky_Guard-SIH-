from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .schemas import NUMERIC_FIELDS, ValidationReport, observations_to_frame


DEFAULT_RANGES = {
    "temperature": (-50.0, 65.0),
    "humidity": (-5.0, 105.0),
    "pressure": (850.0, 1100.0),
    "rainfall": (0.0, 2000.0),
    "wind_speed": (0.0, 150.0),
    "wind_direction": (0.0, 360.0),
    "latitude": (-90.0, 90.0),
    "longitude": (-180.0, 180.0),
}


def validate_observations(
    observations: list[dict[str, Any]] | pd.DataFrame,
    expected_interval_minutes: float | None = None,
    allowed_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, ValidationReport]:
    frame = observations_to_frame(observations)
    flags: dict[int, list[str]] = {}
    summary: Counter[str] = Counter()

    if frame.empty:
        report = ValidationReport(0, 0, 0, {}, {})
        return frame, report

    original_order = frame.index.to_list()

    def add_flag(row_index: int, flag: str) -> None:
        flags.setdefault(row_index, []).append(flag)
        summary[flag] += 1

    for index, row in frame.iterrows():
        if pd.isna(row["station_id"]):
            add_flag(index, "MISSING_STATION_ID")

        if pd.isna(row["timestamp"]):
            add_flag(index, "INVALID_TIMESTAMP")
        elif row["timestamp"] > pd.Timestamp.now(tz="UTC"):
            add_flag(index, "FUTURE_TIMESTAMP")

        for field_name in NUMERIC_FIELDS:
            value = row[field_name]
            if pd.isna(value):
                add_flag(index, f"MISSING_{field_name.upper()}")
            elif not np.isfinite(value):
                add_flag(index, f"INVALID_{field_name.upper()}")

    ranges = dict(DEFAULT_RANGES)
    if allowed_ranges:
        ranges.update(allowed_ranges)

    for field_name, (lower, upper) in ranges.items():
        if field_name not in frame:
            continue

        invalid = frame[field_name].notna() & (
            (frame[field_name] < lower) | (frame[field_name] > upper)
        )

        for index in frame.index[invalid]:
            add_flag(index, f"OUT_OF_RANGE_{field_name.upper()}")

    duplicate_mask = frame.duplicated(
        subset=["station_id", "timestamp"],
        keep=False,
    )

    for index in frame.index[duplicate_mask]:
        add_flag(index, "DUPLICATE_TIMESTAMP")

    duplicate_values = frame.duplicated(keep=False)
    for index in frame.index[duplicate_values]:
        add_flag(index, "DUPLICATE_OBSERVATION")

    for station_id, group in frame.groupby("station_id", dropna=False):
        timestamps = group["timestamp"]
        valid_timestamps = timestamps.dropna()

        if not valid_timestamps.is_monotonic_increasing:
            for index in group.index:
                add_flag(index, "OUT_OF_ORDER")

        if expected_interval_minutes and len(valid_timestamps) > 1:
            ordered = valid_timestamps.sort_values()
            gaps = ordered.diff().dt.total_seconds().div(60.0)
            abnormal = gaps > expected_interval_minutes * 2.5

            for index in ordered.index[abnormal.fillna(False)]:
                add_flag(index, "SAMPLING_GAP")

    frame["quality_flags"] = [
        flags.get(index, [])
        for index in original_order
    ]

    frame["is_valid"] = frame["quality_flags"].map(len).eq(0)

    report = ValidationReport(
        row_count=len(frame),
        valid_row_count=int(frame["is_valid"].sum()),
        invalid_row_count=int((~frame["is_valid"]).sum()),
        quality_flags=flags,
        summary=dict(summary),
    )

    return frame, report
