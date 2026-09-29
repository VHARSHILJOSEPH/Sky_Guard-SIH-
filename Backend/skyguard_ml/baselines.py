from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FeatureConfig


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "post_monsoon"


@dataclass
class BaselineStats:
    count: int
    mean: float
    median: float
    std: float
    mad: float
    q01: float
    q05: float
    q95: float
    q99: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "mad": self.mad,
            "q01": self.q01,
            "q05": self.q05,
            "q95": self.q95,
            "q99": self.q99,
        }


def calculate_stats(values: pd.Series) -> BaselineStats:
    clean = pd.to_numeric(values, errors="coerce").dropna()

    if clean.empty:
        return BaselineStats(
            count=0,
            mean=0.0,
            median=0.0,
            std=0.0,
            mad=0.0,
            q01=0.0,
            q05=0.0,
            q95=0.0,
            q99=0.0,
        )

    median = float(clean.median())
    mad = float(np.median(np.abs(clean - median)))

    return BaselineStats(
        count=len(clean),
        mean=float(clean.mean()),
        median=median,
        std=float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
        mad=mad,
        q01=float(clean.quantile(0.01)),
        q05=float(clean.quantile(0.05)),
        q95=float(clean.quantile(0.95)),
        q99=float(clean.quantile(0.99)),
    )


class BaselineStore:
    def __init__(self, minimum_samples: int = 8) -> None:
        self.minimum_samples = minimum_samples
        self.stats: dict[str, dict[str, dict[str, Any]]] = {}

    def fit(
        self,
        frame: pd.DataFrame,
        variables: list[str],
    ) -> "BaselineStore":
        work = frame.copy()
        work["hour"] = work["timestamp"].dt.hour
        work["season"] = work["timestamp"].dt.month.map(
            season_for_month
        )

        for variable in variables:
            self.stats[variable] = {
                "station_hour_season": {},
                "station_hour": {},
                "station": {},
                "global": calculate_stats(work[variable]).as_dict(),
            }

            for keys, group in work.groupby(
                ["station_id", "hour", "season"],
                dropna=False,
            ):
                station_id, hour, season = keys
                stats = calculate_stats(group[variable])

                if stats.count >= self.minimum_samples:
                    self.stats[variable]["station_hour_season"][
                        f"{station_id}|{hour}|{season}"
                    ] = stats.as_dict()

            for keys, group in work.groupby(
                ["station_id", "hour"],
                dropna=False,
            ):
                station_id, hour = keys
                stats = calculate_stats(group[variable])

                if stats.count >= self.minimum_samples:
                    self.stats[variable]["station_hour"][
                        f"{station_id}|{hour}"
                    ] = stats.as_dict()

            for station_id, group in work.groupby(
                ["station_id"],
                dropna=False,
            ):
                stats = calculate_stats(group[variable])

                if stats.count >= self.minimum_samples:
                    self.stats[variable]["station"][
                        str(station_id)
                    ] = stats.as_dict()

        return self

    def get(
        self,
        variable: str,
        station_id: Any,
        timestamp: pd.Timestamp,
    ) -> tuple[dict[str, Any], str]:
        variable_stats = self.stats.get(variable, {})

        hour = timestamp.hour
        season = season_for_month(timestamp.month)

        candidates = [
            (
                f"{station_id}|{hour}|{season}",
                variable_stats.get("station_hour_season", {}),
                "STATION_HOUR_SEASON",
            ),
            (
                f"{station_id}|{hour}",
                variable_stats.get("station_hour", {}),
                "STATION_HOUR",
            ),
            (
                str(station_id),
                variable_stats.get("station", {}),
                "STATION",
            ),
        ]

        for key, source, source_name in candidates:
            if key in source:
                return source[key], source_name

        return (
            variable_stats.get(
                "global",
                calculate_stats(pd.Series(dtype=float)).as_dict(),
            ),
            "GLOBAL",
        )

    def score(
        self,
        variable: str,
        value: float,
        station_id: Any,
        timestamp: pd.Timestamp,
    ) -> tuple[float, str, dict[str, Any]]:
        if pd.isna(value):
            return 0.0, "UNAVAILABLE", {}

        stats, source = self.get(variable, station_id, timestamp)

        scale = max(
            float(stats.get("mad", 0.0)) * 1.4826,
            float(stats.get("std", 0.0)),
            1e-6,
        )

        robust_z = abs(float(value) - float(stats["median"])) / scale
        score = float(np.clip(robust_z / 6.0, 0.0, 1.0))

        return score, source, stats
