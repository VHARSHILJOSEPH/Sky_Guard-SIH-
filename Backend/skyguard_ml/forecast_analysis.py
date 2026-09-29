from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd



def analyze_forecast_context(
    observation: dict[str, Any],
    forecast_context: dict[str, Any] | None,
    variables: list[str],
    max_freshness_seconds: float = 10800.0,  # 3 hours
) -> dict[str, Any]:
    unavailable_result = {
        "status": "UNAVAILABLE",
        "forecast_score": None,
        "details": {},
        "provenance": "EXTERNAL_FORECAST",
    }

    if forecast_context is None or not forecast_context:
        return unavailable_result

    # Freshness verification
    obs_ts = pd.to_datetime(observation.get("timestamp"), utc=True, errors="coerce")
    fc_ts_raw = forecast_context.get("timestamp") or forecast_context.get("forecast_time")
    if pd.notna(obs_ts) and fc_ts_raw is not None:
        fc_ts = pd.to_datetime(fc_ts_raw, utc=True, errors="coerce")
        if pd.notna(fc_ts):
            delta_sec = abs((obs_ts - fc_ts).total_seconds())
            if delta_sec > max_freshness_seconds:
                return {
                    "status": "UNAVAILABLE",
                    "forecast_score": None,
                    "details": {"reason": f"Forecast timestamp stale by {delta_sec:.0f}s (> {max_freshness_seconds:.0f}s)"},
                    "provenance": "EXTERNAL_FORECAST",
                }

    deviations: list[float] = []
    details: dict[str, Any] = {}

    for variable in variables:
        observed = observation.get(variable)
        forecast = forecast_context.get(variable)

        if observed is None or forecast is None:
            continue

        try:
            observed = float(observed)
            forecast_value = (
                float(forecast["value"])
                if isinstance(forecast, dict)
                else float(forecast)
            )
        except (TypeError, ValueError, KeyError):
            continue

        absolute_deviation = abs(observed - forecast_value)
        denominator = max(abs(forecast_value), 1.0)
        relative_deviation = absolute_deviation / denominator

        range_deviation = 0.0
        forecast_range = None

        if isinstance(forecast, dict):
            lower = forecast.get("min")
            upper = forecast.get("max")

            if lower is not None and upper is not None:
                lower = float(lower)
                upper = float(upper)
                forecast_range = [lower, upper]

                if observed < lower:
                    range_deviation = lower - observed
                elif observed > upper:
                    range_deviation = observed - upper

        score = min(
            1.0,
            max(
                relative_deviation * 3.0,
                range_deviation / max(abs(forecast_value), 1.0),
            ),
        )

        deviations.append(score)
        details[variable] = {
            "observed_value": observed,
            "forecast_value": forecast_value,
            "absolute_deviation": absolute_deviation,
            "relative_deviation": relative_deviation,
            "forecast_range": forecast_range,
            "range_deviation": range_deviation,
            "score": score,
        }

    if not deviations:
        return unavailable_result

    return {
        "status": "AVAILABLE",
        "forecast_score": float(np.mean(deviations)),
        "details": details,
        "provenance": "EXTERNAL_FORECAST",
    }

