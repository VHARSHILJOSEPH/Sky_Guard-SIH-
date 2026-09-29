from __future__ import annotations

from typing import Any


def calculate_selection_score(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> float:
    base = (
        weights.get("f1", 0.30) * metrics.get("f1", 0.0)
        + weights.get("recall", 0.15) * metrics.get("recall", 0.0)
        + weights.get("precision", 0.15) * metrics.get("precision", 0.0)
        + weights.get("balanced_accuracy", 0.20) * metrics.get("balanced_accuracy", 0.0)
        + weights.get("specificity", 0.20) * metrics.get("specificity", 0.0)
        - weights.get("false_positive_rate", 0.25) * metrics.get("false_positive_rate", 0.0)
        - weights.get("false_alarm_rate", 0.10) * min(metrics.get("false_alarms_per_station_day", 0.0) / 24.0, 1.0)
    )
    return float(base)


def select_best_model(
    model_results: dict[str, dict[str, Any]],
    weights: dict[str, float],
) -> tuple[str, dict[str, Any]]:
    if not model_results:
        raise ValueError("No valid model results were provided.")

    best_name = None
    best_result = None
    best_score = float("-inf")

    for name, result in model_results.items():
        metrics = result.get("validation_metrics", {})
        score = calculate_selection_score(metrics, weights)
        result["selection_score"] = score

        if score > best_score:
            best_score = score
            best_name = name
            best_result = result

    if best_name is None or best_result is None:
        raise ValueError("Unable to select a model.")

    return best_name, best_result
