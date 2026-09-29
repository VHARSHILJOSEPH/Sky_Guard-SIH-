from __future__ import annotations

import pandas as pd


def sort_by_station_time(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(
        result["timestamp"],
        utc=True,
        errors="coerce",
    )

    return result.sort_values(
        ["station_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = sort_by_station_time(frame)

    if ordered.empty:
        return ordered.copy(), ordered.copy(), ordered.copy()

    timestamps = ordered["timestamp"].dropna().sort_values().unique()

    if len(timestamps) < 3:
        first = int(len(ordered) * train_fraction)
        second = int(
            len(ordered) * (train_fraction + validation_fraction)
        )
        return (
            ordered.iloc[:first].copy(),
            ordered.iloc[first:second].copy(),
            ordered.iloc[second:].copy(),
        )

    train_cutoff = timestamps[
        min(
            len(timestamps) - 1,
            max(0, int(len(timestamps) * train_fraction)),
        )
    ]

    validation_cutoff = timestamps[
        min(
            len(timestamps) - 1,
            max(
                0,
                int(
                    len(timestamps)
                    * (train_fraction + validation_fraction)
                ),
            ),
        )
    ]

    train = ordered[ordered["timestamp"] < train_cutoff]
    validation = ordered[
        (ordered["timestamp"] >= train_cutoff)
        & (ordered["timestamp"] < validation_cutoff)
    ]
    test = ordered[ordered["timestamp"] >= validation_cutoff]

    return train.copy(), validation.copy(), test.copy()


def safe_numeric_frame(
    frame: pd.DataFrame,
    feature_list: list[str],
) -> pd.DataFrame:
    matrix = frame.reindex(columns=feature_list).copy()

    for column in matrix.columns:
        matrix[column] = pd.to_numeric(
            matrix[column],
            errors="coerce",
        )

    matrix = matrix.replace([float("inf"), float("-inf")], pd.NA)
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    matrix = matrix.fillna(0.0)

    return matrix.astype(float)
