from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

from .baselines import BaselineStore, season_for_month
from .config import FeatureConfig
from .preprocessing import sort_by_station_time


# ── SIH core AWS inputs ────────────────────────────────────────────────────
# Temperature, Pressure, Humidity are the mandatory SIH core parameters.
# Wind and rainfall remain in the schema for optional context (statistical
# detectors, validation) but are NOT part of the ML feature vector.
CORE_ML_VARIABLES = [
    "temperature",
    "humidity",
    "pressure",
]

# Full set used by statistical detectors and validation (non-ML)
BASE_VARIABLES = [
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
]


# ── Indian meteorological season mapping ───────────────────────────────────
MONTH_TO_SEASON_CODE = {
    12: 0, 1: 0, 2: 0,       # Winter
    3: 1, 4: 1, 5: 1,        # Summer (Pre-monsoon)
    6: 2, 7: 2, 8: 2, 9: 2,  # Monsoon
    10: 3, 11: 3,             # Post-monsoon
}


@dataclass
class FeatureBuilder:
    config: FeatureConfig
    baseline: BaselineStore | None = None
    feature_list: list[str] = field(default_factory=list)

    def fit(
        self,
        frame: pd.DataFrame,
        baseline: BaselineStore | None = None,
    ) -> "FeatureBuilder":
        self.baseline = baseline
        transformed = self._transform(frame)

        excluded = {
            "station_id",
            "timestamp",
            "quality_flags",
            "is_valid",
            "target",
            "fault_type",
        }

        self.feature_list = [
            column
            for column in transformed.columns
            if column not in excluded
            and pd.api.types.is_numeric_dtype(transformed[column])
        ]

        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return self._transform(frame)

    def fit_transform(
        self,
        frame: pd.DataFrame,
        baseline: BaselineStore | None = None,
    ) -> pd.DataFrame:
        self.fit(frame, baseline)
        return self.transform(frame)

    def _transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        work = sort_by_station_time(frame)

        if work.empty:
            return work

        pieces: list[pd.DataFrame] = []

        for station_id, group in work.groupby(
            "station_id",
            dropna=False,
            sort=False,
        ):
            group = group.copy()

            # ── Temporal cyclic features ────────────────────────────────
            group["hour"] = group["timestamp"].dt.hour
            group["day_of_year"] = group["timestamp"].dt.dayofyear
            group["month"] = group["timestamp"].dt.month

            # Fixed: map integer month -> integer season code
            group["season_code"] = group["month"].map(
                MONTH_TO_SEASON_CODE
            )

            group["hour_sin"] = np.sin(
                2.0 * np.pi * group["hour"] / 24.0
            )
            group["hour_cos"] = np.cos(
                2.0 * np.pi * group["hour"] / 24.0
            )
            group["day_of_year_sin"] = np.sin(
                2.0 * np.pi * group["day_of_year"] / 365.25
            )
            group["day_of_year_cos"] = np.cos(
                2.0 * np.pi * group["day_of_year"] / 365.25
            )

            # ── Elapsed time between observations ──────────────────────
            elapsed = (
                group["timestamp"]
                .diff()
                .dt.total_seconds()
                .div(3600.0)
            )
            group["elapsed_hours"] = elapsed.replace(0.0, np.nan)

            # ── Per-variable causal features (CORE_ML_VARIABLES only) ──
            for variable in CORE_ML_VARIABLES:
                if variable not in group:
                    continue

                # Delta and rate — strictly causal (shift 1)
                previous = group[variable].shift(1)
                group[f"{variable}_delta"] = group[variable] - previous
                group[f"{variable}_rate"] = (
                    group[f"{variable}_delta"] / group["elapsed_hours"]
                )

                # Lag features — strictly causal
                for lag in (1, 2, 3):
                    group[f"{variable}_lag_{lag}"] = group[
                        variable
                    ].shift(lag)

                # Rolling statistics — strictly causal: shift(1) BEFORE
                # rolling so the current observation is excluded.
                shifted = group[variable].shift(1)

                for window in self.config.rolling_windows:
                    rolling = shifted.rolling(
                        window=window,
                        min_periods=max(2, window // 2),
                    )

                    group[f"{variable}_rolling_mean_{window}"] = (
                        rolling.mean()
                    )
                    group[f"{variable}_rolling_std_{window}"] = (
                        rolling.std()
                    )

                # Historical baseline score
                if self.baseline is not None:
                    scores: list[float] = []

                    for _, row in group.iterrows():
                        score, _, _ = self.baseline.score(
                            variable=variable,
                            value=row[variable],
                            station_id=row["station_id"],
                            timestamp=row["timestamp"],
                        )
                        scores.append(score)

                    group[f"{variable}_historical_score"] = scores

            # ── Multivariate interaction features ──────────────────────
            # Fixed: use shift(1) before rolling to prevent target leakage.
            temperature = group.get("temperature")
            humidity = group.get("humidity")
            pressure = group.get("pressure")

            if temperature is not None and humidity is not None:
                temp_shifted_mean = temperature.shift(1).rolling(
                    6, min_periods=2
                ).mean()
                hum_shifted_mean = humidity.shift(1).rolling(
                    6, min_periods=2
                ).mean()

                group["temperature_humidity_interaction"] = (
                    temperature * humidity / 100.0
                )
                group["temperature_humidity_deviation"] = (
                    temperature - temp_shifted_mean
                ) * (
                    humidity - hum_shifted_mean
                )

            if temperature is not None and pressure is not None:
                temp_shifted_mean = temperature.shift(1).rolling(
                    6, min_periods=2
                ).mean()
                pres_shifted_mean = pressure.shift(1).rolling(
                    6, min_periods=2
                ).mean()

                group["temperature_pressure_deviation"] = (
                    temperature - temp_shifted_mean
                ) * (
                    pressure - pres_shifted_mean
                )

            if humidity is not None and pressure is not None:
                hum_shifted_mean = humidity.shift(1).rolling(
                    6, min_periods=2
                ).mean()
                pres_shifted_mean = pressure.shift(1).rolling(
                    6, min_periods=2
                ).mean()

                group["humidity_pressure_deviation"] = (
                    humidity - hum_shifted_mean
                ) * (
                    pressure - pres_shifted_mean
                )

            pieces.append(group)

        return pd.concat(pieces, ignore_index=True)
