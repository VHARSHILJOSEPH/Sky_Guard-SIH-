from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _bounded(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _score_from_magnitude(
    magnitude: float,
    threshold: float,
    saturation: float,
) -> float:
    if not np.isfinite(magnitude):
        return 0.0

    if magnitude <= threshold:
        return 0.0

    return _bounded(
        (magnitude - threshold)
        / max(saturation - threshold, 1e-6)
    )


def detect_spike_drop(
    row: pd.Series,
    variables: list[str],
    spike_thresholds: dict[str, float] | None = None,
) -> dict[str, dict[str, Any]]:
    thresholds = spike_thresholds or {
        "temperature": 3.0,
        "humidity": 15.0,
        "pressure": 4.0,
        "rainfall": 10.0,
        "wind_speed": 15.0,
    }

    results: dict[str, dict[str, Any]] = {}

    for variable in variables:
        delta = row.get(f"{variable}_delta", np.nan)
        threshold = thresholds.get(variable, 1.0)

        results[variable] = {
            "spike_score": _score_from_magnitude(
                max(float(delta), 0.0)
                if pd.notna(delta)
                else np.nan,
                threshold,
                threshold * 3.0,
            ),
            "drop_score": _score_from_magnitude(
                max(-float(delta), 0.0)
                if pd.notna(delta)
                else np.nan,
                threshold,
                threshold * 3.0,
            ),
            "delta": None if pd.isna(delta) else float(delta),
        }

    return results


def detect_drift(
    history: pd.DataFrame,
    variables: list[str],
    window: int = 12,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    if len(history) < 4:
        return {
            variable: {
                "score": 0.0,
                "slope": None,
                "status": "INSUFFICIENT_DATA",
            }
            for variable in variables
        }

    recent = history.tail(window).copy()

    for variable in variables:
        if variable not in recent:
            continue

        values = pd.to_numeric(
            recent[variable],
            errors="coerce",
        ).dropna()

        if len(values) < 4:
            result[variable] = {
                "score": 0.0,
                "slope": None,
                "status": "INSUFFICIENT_DATA",
            }
            continue

        x = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x, values.to_numpy(), 1)[0])

        scale = max(float(values.std()), 1e-6)
        standardized_slope = abs(slope) * len(values) / scale

        result[variable] = {
            "score": _bounded(standardized_slope / 4.0),
            "slope": slope,
            "status": "AVAILABLE",
        }

    return result


def detect_frozen_sensor(
    history: pd.DataFrame,
    variables: list[str],
    tolerance: float = 1e-4,
    duration_hours: float = 2.0,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}

    for variable in variables:
        if variable not in history or history.empty:
            result[variable] = {
                "score": 0.0,
                "run_length": 0,
                "status": "INSUFFICIENT_DATA",
            }
            continue

        values = pd.to_numeric(
            history[variable],
            errors="coerce",
        ).dropna()

        if len(values) < 3:
            result[variable] = {
                "score": 0.0,
                "run_length": int(len(values)),
                "status": "INSUFFICIENT_DATA",
            }
            continue

        latest = float(values.iloc[-1])

        # Zero rainfall is normal dry atmospheric condition, not a frozen instrument
        if variable == "rainfall" and abs(latest) <= tolerance:
            result[variable] = {
                "score": 0.0,
                "run_length": 0,
                "duration_hours": 0.0,
                "variance": 0.0,
                "status": "DRY_CONDITIONS",
            }
            continue

        run_length = 1

        for value in reversed(values.iloc[:-1].to_numpy()):
            if abs(float(value) - latest) <= tolerance:
                run_length += 1
            else:
                break

        timestamps = pd.to_datetime(
            history["timestamp"],
            utc=True,
            errors="coerce",
        ).dropna()

        duration = 0.0
        if len(timestamps) >= 2:
            duration = float(
                (
                    timestamps.iloc[-1] - timestamps.iloc[-run_length]
                ).total_seconds()
                / 3600.0
            )

        # Calculate variance across recent run window
        recent_window = values.iloc[-run_length:].to_numpy()
        variance = float(np.var(recent_window)) if len(recent_window) > 1 else 1.0

        score = 0.0
        if duration >= duration_hours:
            score = _bounded(duration / (duration_hours * 2.5))

        # Flag when variance is 0.000 across >= 6 consecutive readings
        if run_length >= 6 and variance <= tolerance:
            score = max(score, min(1.0, 0.7 + (run_length - 6) * 0.1))

        result[variable] = {
            "score": score,
            "run_length": run_length,
            "duration_hours": duration,
            "variance": variance,
            "status": "AVAILABLE",
        }

    return result


def detect_missing_data(
    history: pd.DataFrame,
    expected_interval_minutes: float | None = None,
) -> dict[str, Any]:
    if history.empty:
        return {
            "score": 0.0,
            "missing_count": 0,
            "gap_count": 0,
            "status": "INSUFFICIENT_DATA",
        }

    core_cols = [c for c in ["temperature", "humidity", "pressure"] if c in history.columns]
    missing_count = int(history[core_cols].isna().any(axis=1).sum()) if core_cols else 0

    timestamps = pd.to_datetime(
        history["timestamp"],
        utc=True,
        errors="coerce",
    ).dropna().sort_values()

    gap_count = 0
    if expected_interval_minutes and len(timestamps) > 1:
        gaps = timestamps.diff().dt.total_seconds().div(60.0)
        gap_count = int(
            (gaps > expected_interval_minutes * 2.5).sum()
        )

    score = min(
        1.0,
        missing_count / max(len(history), 1)
        + gap_count / max(len(history), 1),
    )

    return {
        "score": float(score),
        "missing_count": missing_count,
        "gap_count": gap_count,
        "status": "AVAILABLE",
    }


def run_statistical_detectors(
    history: pd.DataFrame,
    current_features: pd.Series,
    variables: list[str],
    frozen_tolerance: float = 1e-4,
    frozen_duration_hours: float = 2.0,
    expected_interval_minutes: float | None = None,
) -> dict[str, Any]:
    spike_drop = detect_spike_drop(
        current_features,
        variables,
    )
    drift = detect_drift(history, variables)
    frozen = detect_frozen_sensor(
        history,
        variables,
        tolerance=frozen_tolerance,
        duration_hours=frozen_duration_hours,
    )
    missing = detect_missing_data(
        history,
        expected_interval_minutes=expected_interval_minutes,
    )

    spike_scores = [
        item["spike_score"]
        for item in spike_drop.values()
    ]
    drop_scores = [
        item["drop_score"]
        for item in spike_drop.values()
    ]
    drift_scores = [
        item["score"]
        for item in drift.values()
    ]
    frozen_scores = [
        item["score"]
        for item in frozen.values()
    ]

    temporal_score = max(
        spike_scores + drop_scores + drift_scores + frozen_scores + [0.0]
    )

    return {
        "spike_drop": spike_drop,
        "drift": drift,
        "frozen": frozen,
        "missing": missing,
        "temporal_score": float(temporal_score),
    }
