from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    earth_radius = 6371.0

    lat1_rad, lat2_rad = np.radians([lat1, lat2])
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad)
        * np.cos(lat2_rad)
        * np.sin(delta_lon / 2.0) ** 2
    )

    return float(
        2.0 * earth_radius * np.arcsin(np.sqrt(a))
    )


def analyze_spatial_context(
    observation: dict[str, Any],
    nearby_context: list[dict[str, Any]] | pd.DataFrame | None,
    variables: list[str],
    radius_km: float = 100.0,
    max_freshness_seconds: float = 10800.0,  # 3 hours
) -> dict[str, Any]:
    unavailable_result = {
        "status": "SPATIAL_EVIDENCE_UNAVAILABLE",
        "spatial_score": None,
        "weather_event_evidence": None,
        "sensor_disagreement_evidence": None,
        "spatial_agreement_score": None,
        "number_of_nearby_stations": 0,
        "fresh_station_count": 0,
        "stale_station_count": 0,
    }

    if nearby_context is None:
        return unavailable_result

    if isinstance(nearby_context, pd.DataFrame):
        nearby = nearby_context.copy()
    else:
        nearby = pd.DataFrame(nearby_context)

    if nearby.empty:
        return unavailable_result

    target_lat = observation.get("latitude")
    target_lon = observation.get("longitude")

    if pd.isna(target_lat) or pd.isna(target_lon):
        return unavailable_result

    # Freshness verification
    obs_ts = pd.to_datetime(observation.get("timestamp"), utc=True, errors="coerce")
    stale_count = 0
    fresh_rows = []

    for _, row in nearby.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
            continue

        dist = haversine_km(
            float(target_lat),
            float(target_lon),
            float(row["latitude"]),
            float(row["longitude"]),
        )
        if dist > radius_km:
            continue

        # Check timestamp freshness if observation timestamp is known
        if pd.notna(obs_ts) and "timestamp" in row and pd.notna(row["timestamp"]):
            row_ts = pd.to_datetime(row["timestamp"], utc=True, errors="coerce")
            if pd.notna(row_ts):
                delta_sec = abs((obs_ts - row_ts).total_seconds())
                if delta_sec > max_freshness_seconds:
                    stale_count += 1
                    continue

        row_dict = dict(row)
        row_dict["distance_km"] = dist
        fresh_rows.append(row_dict)

    if not fresh_rows:
        unavailable_result["stale_station_count"] = stale_count
        return unavailable_result

    fresh_df = pd.DataFrame(fresh_rows)
    result: dict[str, Any] = {
        "number_of_nearby_stations": int(len(fresh_df)),
        "fresh_station_count": int(len(fresh_df)),
        "stale_station_count": stale_count,
    }

    deviations: list[float] = []
    similar_count = 0

    for variable in variables:
        values = pd.to_numeric(
            fresh_df.get(variable),
            errors="coerce",
        ).dropna()

        observed = observation.get(variable)

        if values.empty or observed is None or pd.isna(observed):
            continue

        mean = float(values.mean())
        median = float(values.median())
        std = float(values.std()) if len(values) > 1 else 0.0

        deviation = abs(float(observed) - median)
        scale = max(std, 1e-6)

        deviations.append(deviation / scale)

        if deviation <= max(scale * 2.0, 1.0):
            similar_count += 1

        result[f"{variable}_nearby_mean"] = mean
        result[f"{variable}_nearby_median"] = median
        result[f"{variable}_nearby_std"] = std

    if not deviations:
        return unavailable_result

    normalized_deviation = min(
        1.0,
        float(np.mean(deviations)) / 5.0,
    )
    agreement = similar_count / max(len(variables), 1)

    result["spatial_score"] = normalized_deviation
    result["weather_event_evidence"] = float(agreement)
    result["sensor_disagreement_evidence"] = float(1.0 - agreement)
    result["number_showing_similar_change"] = similar_count
    result["spatial_agreement_score"] = float(agreement)

    # Standardize to canonical 3 states
    if agreement >= 0.50:
        result["status"] = "SPATIAL_CORROBORATED"
    else:
        result["status"] = "SPATIAL_NOT_CORROBORATED"

    return result

