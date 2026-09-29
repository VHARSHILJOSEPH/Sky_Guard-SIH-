I’ll provide a complete from-scratch reference implementation as a modular Python package, not a toy classifier. The code will keep training, evaluation, inference, evidence fusion, artifact persistence, and tests separate so you can adapt each part to your ESP32/API data later.The package below is a complete reference implementation for the SkyGuard ML backend engine. It intentionally does not include frontend, FastAPI, database, ESP32 firmware, authentication, forecasting, or deployment code.

It uses:

- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- Robust Covariance when valid
- Chronological validation
- Synthetic labelled faults
- Historical station baselines
- Forecast evidence
- Spatial corroboration
- Statistical detectors
- Sensor-health tracking
- Explainable output
- Persisted model artifacts

## Project Structure

```text
skyguard-project/
├── pyproject.toml
├── skyguard_ml/
│   ├── __init__.py
│   ├── schemas.py
│   ├── config.py
│   ├── validation.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── baselines.py
│   ├── statistical_detectors.py
│   ├── synthetic_faults.py
│   ├── spatial_analysis.py
│   ├── forecast_analysis.py
│   ├── evidence_fusion.py
│   ├── event_context.py
│   ├── anomaly_types.py
│   ├── sensor_health.py
│   ├── explainability.py
│   ├── model_selection.py
│   ├── training.py
│   ├── evaluation.py
│   ├── inference.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── isolation_forest.py
│   │   ├── lof.py
│   │   ├── one_class_svm.py
│   │   ├── robust_covariance.py
│   │   └── autoencoder.py
│   └── artifacts/
└── tests/
    ├── test_validation.py
    ├── test_features.py
    ├── test_detectors.py
    └── test_end_to_end.py
```

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "skyguard-ml"
version = "0.1.0"
description = "Weather sensor anomaly detection engine"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "joblib>=1.3"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4"
]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

## `skyguard_ml/schemas.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd


OBSERVATION_FIELDS = [
    "station_id",
    "timestamp",
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "latitude",
    "longitude",
]

NUMERIC_FIELDS = [
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "latitude",
    "longitude",
]


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def normalize_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "temp": "temperature",
        "temperature_c": "temperature",
        "relative_humidity": "humidity",
        "humidity_percent": "humidity",
        "pressure_hpa": "pressure",
        "rain": "rainfall",
        "rainfall_mm": "rainfall",
        "wind": "wind_speed",
        "wind_speed_kmh": "wind_speed",
    }

    normalized: dict[str, Any] = {}

    for key, value in observation.items():
        normalized[aliases.get(key, key)] = value

    normalized.setdefault("station_id", None)
    normalized.setdefault("timestamp", None)

    normalized["timestamp"] = normalize_timestamp(normalized["timestamp"])

    for field_name in NUMERIC_FIELDS:
        value = normalized.get(field_name)
        if value is None or value == "":
            normalized[field_name] = np.nan
            continue

        try:
            normalized[field_name] = float(value)
        except (TypeError, ValueError):
            normalized[field_name] = np.nan

    return {
        field_name: normalized.get(field_name, np.nan)
        for field_name in OBSERVATION_FIELDS
    }


def observations_to_frame(
    observations: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    if isinstance(observations, pd.DataFrame):
        records = observations.to_dict(orient="records")
    else:
        records = observations

    normalized = [normalize_observation(record) for record in records]
    frame = pd.DataFrame(normalized)

    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"],
            utc=True,
            errors="coerce",
        )

    return frame


@dataclass
class ValidationReport:
    row_count: int
    valid_row_count: int
    invalid_row_count: int
    quality_flags: dict[int, list[str]] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        if self.row_count == 0:
            return 0.0
        return float(self.valid_row_count / self.row_count)


@dataclass
class DetectorEvidence:
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    status: str
    anomaly_score: float
    selected_model: str
    anomaly_type: str
    severity: str
    event_classification: str
    evidence: dict[str, Any]
    sensor_health: dict[str, Any]
    reason_codes: list[str]
    explanation: str
    recommended_action: str
    confidence_status: str = "NOT_CALIBRATED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "anomaly_score": round(float(self.anomaly_score), 4),
            "selected_model": self.selected_model,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "event_classification": self.event_classification,
            "evidence": self.evidence,
            "sensor_health": self.sensor_health,
            "reason_codes": self.reason_codes,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "confidence_status": self.confidence_status,
        }
```

## `skyguard_ml/config.py`

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json
from pathlib import Path


@dataclass
class FusionWeights:
    data_quality: float = 0.10
    statistical: float = 0.20
    temporal: float = 0.15
    ml: float = 0.25
    multivariate: float = 0.10
    spatial: float = 0.10
    forecast: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class SelectionWeights:
    f1: float = 0.35
    recall: float = 0.25
    precision: float = 0.20
    false_positive_rate: float = 0.10
    false_alarm_rate: float = 0.10

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class FeatureConfig:
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 24)
    rolling_windows: tuple[int, ...] = (3, 6, 12, 24)
    minimum_baseline_samples: int = 8
    frozen_tolerance: float = 1e-4
    frozen_duration_hours: float = 2.0


@dataclass
class TrainingConfig:
    random_state: int = 42
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    synthetic_faults_per_type: int = 12
    min_training_rows: int = 30
    artifact_dir: str = "skyguard_ml/artifacts"
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)
    selection_weights: SelectionWeights = field(
        default_factory=SelectionWeights
    )
    feature: FeatureConfig = field(default_factory=FeatureConfig)

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        Path(path).write_text(json.dumps(payload, indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "TrainingConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text())
        return cls(
            random_state=payload.get("random_state", 42),
            train_fraction=payload.get("train_fraction", 0.60),
            validation_fraction=payload.get("validation_fraction", 0.20),
            test_fraction=payload.get("test_fraction", 0.20),
            synthetic_faults_per_type=payload.get(
                "synthetic_faults_per_type",
                12,
            ),
            min_training_rows=payload.get("min_training_rows", 30),
            artifact_dir=payload.get(
                "artifact_dir",
                "skyguard_ml/artifacts",
            ),
            fusion_weights=FusionWeights(
                **payload.get("fusion_weights", {})
            ),
            selection_weights=SelectionWeights(
                **payload.get("selection_weights", {})
            ),
            feature=FeatureConfig(
                **payload.get("feature", {})
            ),
        )
```

## `skyguard_ml/validation.py`

```python
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from .schemas import NUMERIC_FIELDS, ValidationReport, observations_to_frame


DEFAULT_RANGES = {
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
```

## `skyguard_ml/preprocessing.py`

```python
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
```

## `skyguard_ml/baselines.py`

```python
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
```

## `skyguard_ml/feature_engineering.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import BaselineStore, season_for_month
from .config import FeatureConfig
from .preprocessing import sort_by_station_time


BASE_VARIABLES = [
    "temperature",
    "humidity",
    "pressure",
    "rainfall",
    "wind_speed",
    "wind_direction",
]


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
            group["hour"] = group["timestamp"].dt.hour
            group["day_of_week"] = group["timestamp"].dt.dayofweek
            group["day_of_year"] = group["timestamp"].dt.dayofyear
            group["month"] = group["timestamp"].dt.month
            group["season_code"] = group["month"].map(
                {
                    "winter": 0,
                    "summer": 1,
                    "monsoon": 2,
                    "post_monsoon": 3,
                }
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

            elapsed = (
                group["timestamp"]
                .diff()
                .dt.total_seconds()
                .div(3600.0)
            )
            group["elapsed_hours"] = elapsed.replace(0.0, np.nan)

            for variable in BASE_VARIABLES:
                if variable not in group:
                    continue

                previous = group[variable].shift(1)
                group[f"{variable}_delta"] = group[variable] - previous
                group[f"{variable}_rate"] = (
                    group[f"{variable}_delta"] / group["elapsed_hours"]
                )

                for lag in self.config.lags:
                    group[f"{variable}_lag_{lag}"] = group[
                        variable
                    ].shift(lag)

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
                    group[f"{variable}_rolling_median_{window}"] = (
                        rolling.median()
                    )
                    group[f"{variable}_rolling_min_{window}"] = (
                        rolling.min()
                    )
                    group[f"{variable}_rolling_max_{window}"] = (
                        rolling.max()
                    )

                if self.baseline is not None:
                    scores: list[float] = []
                    sources: list[str] = []

                    for _, row in group.iterrows():
                        score, source, _ = self.baseline.score(
                            variable=variable,
                            value=row[variable],
                            station_id=row["station_id"],
                            timestamp=row["timestamp"],
                        )
                        scores.append(score)
                        sources.append(source)

                    group[f"{variable}_historical_score"] = scores
                    group[f"{variable}_baseline_source"] = sources

            temperature = group.get("temperature")
            humidity = group.get("humidity")
            pressure = group.get("pressure")

            if temperature is not None and humidity is not None:
                group["temperature_humidity_interaction"] = (
                    temperature * humidity / 100.0
                )
                group["temperature_humidity_deviation"] = (
                    temperature
                    - temperature.rolling(6, min_periods=2).mean()
                ) * (
                    humidity
                    - humidity.rolling(6, min_periods=2).mean()
                )

            if temperature is not None and pressure is not None:
                group["temperature_pressure_deviation"] = (
                    temperature
                    - temperature.rolling(6, min_periods=2).mean()
                ) * (
                    pressure
                    - pressure.rolling(6, min_periods=2).mean()
                )

            if humidity is not None and pressure is not None:
                group["humidity_pressure_deviation"] = (
                    humidity
                    - humidity.rolling(6, min_periods=2).mean()
                ) * (
                    pressure
                    - pressure.rolling(6, min_periods=2).mean()
                )

            pieces.append(group)

        return pd.concat(pieces, ignore_index=True)
```

## `skyguard_ml/statistical_detectors.py`

```python
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

        score = 0.0
        if duration >= duration_hours:
            score = _bounded(duration / (duration_hours * 4.0))

        result[variable] = {
            "score": score,
            "run_length": run_length,
            "duration_hours": duration,
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

    missing_count = int(history.isna().any(axis=1).sum())

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
```

## `skyguard_ml/synthetic_faults.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FaultRecord:
    fault_type: str
    start_index: int
    duration: int
    magnitude: float
    direction: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "start_index": self.start_index,
            "duration": self.duration,
            "magnitude": self.magnitude,
            "direction": self.direction,
        }


FAULT_TYPES = [
    "TEMPERATURE_SPIKE",
    "TEMPERATURE_DROP",
    "TEMPERATURE_DRIFT",
    "HUMIDITY_SPIKE",
    "HUMIDITY_DRIFT",
    "PRESSURE_SPIKE",
    "PRESSURE_DRIFT",
    "FROZEN_SENSOR",
    "MISSING_DATA",
    "COMMUNICATION_FAILURE",
    "DUPLICATE_DATA",
    "TIMESTAMP_ERROR",
    "MULTIVARIATE_INCONSISTENCY",
]


def _valid_start(
    length: int,
    duration: int,
    rng: np.random.Generator,
) -> int:
    if length <= duration + 2:
        return 0
    return int(rng.integers(1, length - duration - 1))


def _choose_station_frame(
    frame: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    stations = frame["station_id"].dropna().unique()

    if len(stations) == 0:
        return frame.copy()

    station = rng.choice(stations)
    result = frame[frame["station_id"] == station].copy()

    return result.sort_values("timestamp").reset_index(drop=True)


def generate_synthetic_faults(
    clean_frame: pd.DataFrame,
    faults_per_type: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    generated_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []

    for fault_type in FAULT_TYPES:
        for _ in range(faults_per_type):
            frame = _choose_station_frame(clean_frame, rng)

            if len(frame) < 8:
                continue

            duration = int(
                rng.integers(
                    1,
                    max(2, min(12, len(frame) // 3)),
                )
            )
            start = _valid_start(len(frame), duration, rng)
            magnitude = float(rng.uniform(0.5, 2.0))
            direction = int(rng.choice([-1, 1]))

            modified = frame.copy()
            indices = modified.index[start:start + duration]

            if fault_type == "TEMPERATURE_SPIKE":
                modified.loc[indices, "temperature"] += (
                    rng.uniform(3.0, 10.0)
                )

            elif fault_type == "TEMPERATURE_DROP":
                modified.loc[indices, "temperature"] -= (
                    rng.uniform(3.0, 10.0)
                )

            elif fault_type == "TEMPERATURE_DRIFT":
                modified.loc[indices, "temperature"] += (
                    np.arange(duration)
                    * rng.uniform(0.2, 1.0)
                    * direction
                )

            elif fault_type == "HUMIDITY_SPIKE":
                modified.loc[indices, "humidity"] += (
                    rng.uniform(15.0, 35.0)
                )

            elif fault_type == "HUMIDITY_DRIFT":
                modified.loc[indices, "humidity"] += (
                    np.arange(duration)
                    * rng.uniform(1.0, 4.0)
                    * direction
                )

            elif fault_type == "PRESSURE_SPIKE":
                modified.loc[indices, "pressure"] += (
                    rng.uniform(5.0, 20.0)
                )

            elif fault_type == "PRESSURE_DRIFT":
                modified.loc[indices, "pressure"] += (
                    np.arange(duration)
                    * rng.uniform(0.5, 2.0)
                    * direction
                )

            elif fault_type == "FROZEN_SENSOR":
                if start > 0:
                    modified.loc[indices, "temperature"] = (
                        modified.loc[start - 1, "temperature"]
                    )

            elif fault_type == "MISSING_DATA":
                modified.loc[indices, "temperature"] = np.nan
                modified.loc[indices, "humidity"] = np.nan

            elif fault_type == "COMMUNICATION_FAILURE":
                modified = modified.drop(index=indices).reset_index(
                    drop=True
                )

            elif fault_type == "DUPLICATE_DATA":
                duplicate_rows = modified.loc[indices].copy()
                modified = pd.concat(
                    [
                        modified.iloc[:start],
                        duplicate_rows,
                        modified.iloc[start:],
                    ],
                    ignore_index=True,
                )

            elif fault_type == "TIMESTAMP_ERROR":
                if len(indices) > 0:
                    modified.loc[indices[0], "timestamp"] = (
                        pd.Timestamp("1900-01-01", tz="UTC")
                    )

            elif fault_type == "MULTIVARIATE_INCONSISTENCY":
                modified.loc[indices, "temperature"] += (
                    rng.uniform(5.0, 15.0)
                )
                modified.loc[indices, "humidity"] = (
                    modified.loc[indices, "humidity"]
                    .clip(40.0, 60.0)
                )

            modified["target"] = 0
            modified["fault_type"] = "NORMAL"

            if len(modified) > 0:
                affected_start = min(start, len(modified) - 1)
                affected_end = min(
                    affected_start + duration,
                    len(modified),
                )
                modified.loc[
                    modified.index[affected_start:affected_end],
                    "target",
                ] = 1
                modified.loc[
                    modified.index[affected_start:affected_end],
                    "fault_type",
                ] = fault_type

            generated_frames.append(modified)
            metadata.append(
                FaultRecord(
                    fault_type=fault_type,
                    start_index=start,
                    duration=duration,
                    magnitude=magnitude,
                    direction=direction,
                ).as_dict()
            )

    if not generated_frames:
        return clean_frame.copy(), pd.DataFrame()

    generated = pd.concat(generated_frames, ignore_index=True)
    return generated, pd.DataFrame(metadata)
```

## Model Implementations

### `skyguard_ml/models/isolation_forest.py`

```python
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_isolation_forest(
    n_estimators: int = 300,
    max_samples: str | int = "auto",
    contamination: float = "auto",
    max_features: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                IsolationForest(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    contamination=contamination,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
```

### `skyguard_ml/models/lof.py`

```python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_lof(
    n_neighbors: int = 20,
    contamination: float = "auto",
    metric: str = "minkowski",
) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LocalOutlierFactor(
                    n_neighbors=n_neighbors,
                    contamination=contamination,
                    metric=metric,
                    novelty=True,
                ),
            ),
        ]
    )
```

### `skyguard_ml/models/one_class_svm.py`

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import OneClassSVM


def build_one_class_svm(
    kernel: str = "rbf",
    nu: float = 0.05,
    gamma: str | float = "scale",
) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                OneClassSVM(
                    kernel=kernel,
                    nu=nu,
                    gamma=gamma,
                ),
            ),
        ]
    )
```

### `skyguard_ml/models/robust_covariance.py`

```python
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def build_robust_covariance(
    contamination: float = 0.05,
    support_fraction: float | None = None,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                EllipticEnvelope(
                    contamination=contamination,
                    support_fraction=support_fraction,
                    random_state=random_state,
                ),
            ),
        ]
    )
```

### `skyguard_ml/models/autoencoder.py`

```python
def build_autoencoder(*args, **kwargs):
    raise RuntimeError(
        "Autoencoder is intentionally not enabled. "
        "Use it only after implementing sufficient sequential data, "
        "training, threshold calibration, and validation."
    )
```

### `skyguard_ml/models/__init__.py`

```python
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .isolation_forest import build_isolation_forest
from .lof import build_lof
from .one_class_svm import build_one_class_svm
from .robust_covariance import build_robust_covariance


class DetectorModel:
    def __init__(
        self,
        name: str,
        estimator: Any,
        hyperparameters: dict[str, Any],
    ) -> None:
        self.name = name
        self.estimator = estimator
        self.hyperparameters = hyperparameters
        self.normal_score_low = 0.0
        self.normal_score_high = 1.0

    def fit(self, x: pd.DataFrame) -> "DetectorModel":
        self.estimator.fit(x)

        raw = self._raw_score(x)
        finite = raw[np.isfinite(raw)]

        if len(finite) > 2:
            self.normal_score_low = float(np.quantile(finite, 0.01))
            self.normal_score_high = float(np.quantile(finite, 0.99))

        return self

    def _raw_score(self, x: pd.DataFrame) -> np.ndarray:
        model = self.estimator

        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(x))

        final_model = model[-1]
        return np.asarray(final_model.decision_function(x))

    def score_samples(self, x: pd.DataFrame) -> np.ndarray:
        raw = self._raw_score(x)

        denominator = max(
            self.normal_score_high - self.normal_score_low,
            1e-8,
        )

        normalized_normality = (
            raw - self.normal_score_low
        ) / denominator

        return np.clip(1.0 - normalized_normality, 0.0, 1.0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        predictions = self.estimator.predict(x)
        return np.where(predictions == -1, 1, 0)


def build_candidate_models(
    random_state: int = 42,
    max_rows: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {
        "IsolationForest": [
            {
                "n_estimators": 200,
                "max_samples": "auto",
                "contamination": 0.05,
                "max_features": 1.0,
            },
            {
                "n_estimators": 300,
                "max_samples": "auto",
                "contamination": 0.10,
                "max_features": 0.8,
            },
        ],
        "LOF": [
            {
                "n_neighbors": 15,
                "contamination": 0.05,
                "metric": "minkowski",
            },
            {
                "n_neighbors": 25,
                "contamination": 0.10,
                "metric": "minkowski",
            },
        ],
        "OneClassSVM": [
            {
                "kernel": "rbf",
                "nu": 0.05,
                "gamma": "scale",
            },
            {
                "kernel": "rbf",
                "nu": 0.10,
                "gamma": "scale",
            },
        ],
        "RobustCovariance": [
            {
                "contamination": 0.05,
                "support_fraction": None,
            },
        ],
    }

    return candidates


def create_model(
    name: str,
    params: dict[str, Any],
    random_state: int = 42,
) -> DetectorModel:
    if name == "IsolationForest":
        estimator = build_isolation_forest(
            random_state=random_state,
            **params,
        )
    elif name == "LOF":
        estimator = build_lof(**params)
    elif name == "OneClassSVM":
        estimator = build_one_class_svm(**params)
    elif name == "RobustCovariance":
        estimator = build_robust_covariance(
            random_state=random_state,
            **params,
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    return DetectorModel(
        name=name,
        estimator=estimator,
        hyperparameters=params,
    )
```

## `skyguard_ml/spatial_analysis.py`

```python
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
) -> dict[str, Any]:
    if nearby_context is None:
        return {
            "status": "UNAVAILABLE",
            "spatial_score": None,
            "weather_event_evidence": None,
            "sensor_disagreement_evidence": None,
        }

    if isinstance(nearby_context, pd.DataFrame):
        nearby = nearby_context.copy()
    else:
        nearby = pd.DataFrame(nearby_context)

    if nearby.empty:
        return {
            "status": "UNAVAILABLE",
            "spatial_score": None,
            "weather_event_evidence": None,
            "sensor_disagreement_evidence": None,
        }

    target_lat = observation.get("latitude")
    target_lon = observation.get("longitude")

    if pd.isna(target_lat) or pd.isna(target_lon):
        return {
            "status": "UNAVAILABLE",
            "spatial_score": None,
            "weather_event_evidence": None,
            "sensor_disagreement_evidence": None,
        }

    distances: list[float] = []

    for _, row in nearby.iterrows():
        if pd.isna(row.get("latitude")) or pd.isna(
            row.get("longitude")
        ):
            distances.append(np.inf)
        else:
            distances.append(
                haversine_km(
                    float(target_lat),
                    float(target_lon),
                    float(row["latitude"]),
                    float(row["longitude"]),
                )
            )

    nearby = nearby.copy()
    nearby["distance_km"] = distances
    nearby = nearby[nearby["distance_km"] <= radius_km]

    if nearby.empty:
        return {
            "status": "UNAVAILABLE",
            "spatial_score": None,
            "weather_event_evidence": None,
            "sensor_disagreement_evidence": None,
        }

    result: dict[str, Any] = {
        "status": "AVAILABLE",
        "number_of_nearby_stations": int(len(nearby)),
    }

    deviations: list[float] = []
    similar_count = 0

    for variable in variables:
        values = pd.to_numeric(
            nearby.get(variable),
            errors="coerce",
        ).dropna()

        observed = observation.get(variable)

        if values.empty or pd.isna(observed):
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
        return {
            "status": "UNAVAILABLE",
            "spatial_score": None,
            "weather_event_evidence": None,
            "sensor_disagreement_evidence": None,
        }

    normalized_deviation = min(
        1.0,
        float(np.mean(deviations)) / 5.0,
    )
    agreement = similar_count / max(len(variables), 1)

    result["spatial_score"] = normalized_deviation
    result["weather_event_evidence"] = float(agreement)
    result["sensor_disagreement_evidence"] = float(
        1.0 - agreement
    )
    result["number_showing_similar_change"] = similar_count
    result["spatial_agreement_score"] = float(agreement)

    return result
```

## `skyguard_ml/forecast_analysis.py`

```python
from __future__ import annotations

from typing import Any

import numpy as np


def analyze_forecast_context(
    observation: dict[str, Any],
    forecast_context: dict[str, Any] | None,
    variables: list[str],
) -> dict[str, Any]:
    if forecast_context is None:
        return {
            "status": "UNAVAILABLE",
            "forecast_score": None,
            "details": {},
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
        return {
            "status": "UNAVAILABLE",
            "forecast_score": None,
            "details": {},
        }

    return {
        "status": "AVAILABLE",
        "forecast_score": float(np.mean(deviations)),
        "details": details,
    }
```

## `skyguard_ml/evidence_fusion.py`

```python
from __future__ import annotations

from typing import Any


def fuse_evidence(
    evidence: dict[str, float | None],
    weights: dict[str, float],
) -> dict[str, Any]:
    weighted_sum = 0.0
    total_weight = 0.0
    used_sources: list[str] = []
    unavailable_sources: list[str] = []

    for source, value in evidence.items():
        if value is None:
            unavailable_sources.append(source)
            continue

        numeric_value = max(0.0, min(1.0, float(value)))
        weight = float(weights.get(source, 0.0))

        weighted_sum += numeric_value * weight
        total_weight += weight
        used_sources.append(source)

    if total_weight == 0.0:
        return {
            "score": 0.0,
            "used_sources": [],
            "unavailable_sources": unavailable_sources,
            "status": "INSUFFICIENT_DATA",
        }

    return {
        "score": float(weighted_sum / total_weight),
        "used_sources": used_sources,
        "unavailable_sources": unavailable_sources,
        "status": "AVAILABLE",
    }
```

## `skyguard_ml/anomaly_types.py`

```python
from __future__ import annotations

from typing import Any


def infer_anomaly_type(
    statistical: dict[str, Any],
    quality_flags: list[str],
    multivariate_score: float,
) -> str:
    if any("TIMESTAMP" in flag for flag in quality_flags):
        return "TIMESTAMP_ERROR"

    if any("DUPLICATE" in flag for flag in quality_flags):
        return "DUPLICATE_DATA"

    missing = statistical.get("missing", {})
    if missing.get("gap_count", 0) > 0:
        return "COMMUNICATION_FAILURE"

    if missing.get("missing_count", 0) > 0:
        return "MISSING_DATA"

    for variable, values in statistical.get(
        "frozen",
        {},
    ).items():
        if values.get("score", 0.0) >= 0.6:
            return "FROZEN_SENSOR"

    for variable, values in statistical.get(
        "spike_drop",
        {},
    ).items():
        if values.get("spike_score", 0.0) >= 0.6:
            return f"{variable.upper()}_SPIKE"

        if values.get("drop_score", 0.0) >= 0.6:
            return f"{variable.upper()}_DROP"

    for variable, values in statistical.get(
        "drift",
        {},
    ).items():
        if values.get("score", 0.0) >= 0.6:
            return f"{variable.upper()}_DRIFT"

    if multivariate_score >= 0.6:
        return "MULTIVARIATE_INCONSISTENCY"

    return "UNKNOWN_ANOMALY"
```

## `skyguard_ml/event_context.py`

```python
from __future__ import annotations

from typing import Any


def classify_event_context(
    anomaly_score: float,
    historical_score: float,
    temporal_score: float,
    multivariate_score: float,
    spatial_score: float | None,
    forecast_score: float | None,
) -> str:
    sensor_evidence = (
        0.30 * anomaly_score
        + 0.25 * historical_score
        + 0.20 * temporal_score
        + 0.25 * multivariate_score
    )

    available_weather_sources = [
        value
        for value in [spatial_score, forecast_score]
        if value is not None
    ]

    if available_weather_sources:
        weather_evidence = sum(
            1.0 - value
            for value in available_weather_sources
        ) / len(available_weather_sources)
    else:
        weather_evidence = None

    if anomaly_score < 0.35:
        return "UNCERTAIN"

    if weather_evidence is None:
        if sensor_evidence >= 0.65:
            return "LIKELY_SENSOR_ANOMALY"
        return "UNCERTAIN"

    if sensor_evidence >= 0.65 and weather_evidence <= 0.45:
        return "LIKELY_SENSOR_ANOMALY"

    if weather_evidence >= 0.65 and sensor_evidence <= 0.55:
        return "LIKELY_WEATHER_EVENT"

    return "UNCERTAIN"
```

## `skyguard_ml/sensor_health.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SensorHealthTracker:
    history: dict[str, list[float]] = field(default_factory=dict)
    max_history: int = 100

    def update(
        self,
        station_id: str,
        anomaly_score: float,
        anomaly_type: str,
        missing_score: float,
        drift_score: float,
        spatial_disagreement: float | None,
        forecast_score: float | None,
    ) -> dict[str, Any]:
        values = self.history.setdefault(station_id, [])

        penalty = (
            35.0 * anomaly_score
            + 20.0 * missing_score
            + 15.0 * drift_score
            + 10.0 * (spatial_disagreement or 0.0)
            + 10.0 * (forecast_score or 0.0)
        )

        if anomaly_type in {
            "COMMUNICATION_FAILURE",
            "MISSING_DATA",
            "FROZEN_SENSOR",
        }:
            penalty += 10.0

        score = float(np.clip(100.0 - penalty, 0.0, 100.0))
        values.append(score)

        if len(values) > self.max_history:
            del values[:-self.max_history]

        recent = values[-5:]
        older = values[-10:-5]

        trend = "STABLE"
        status = "HEALTHY"

        if len(older) >= 2 and len(recent) >= 2:
            recent_mean = float(np.mean(recent))
            older_mean = float(np.mean(older))

            if recent_mean < older_mean - 5.0:
                trend = "DECLINING"
            elif recent_mean > older_mean + 5.0:
                trend = "IMPROVING"

        if score < 40:
            status = "CRITICAL"
        elif score < 65:
            status = "DEGRADING"
        elif score < 80:
            status = "WARNING"

        contributors: list[str] = []

        if anomaly_score >= 0.5:
            contributors.append("Repeated or strong anomaly evidence")
        if drift_score >= 0.5:
            contributors.append("Increasing sensor drift")
        if missing_score >= 0.3:
            contributors.append("Missing data or communication gaps")
        if spatial_disagreement is not None:
            if spatial_disagreement >= 0.6:
                contributors.append("Disagreement with nearby stations")
        if forecast_score is not None and forecast_score >= 0.6:
            contributors.append("Repeated forecast disagreement")

        degradation_warning = (
            trend == "DECLINING"
            and len(values) >= 5
            and score < 75
        )

        return {
            "score": round(score, 1),
            "status": status,
            "trend": trend,
            "contributors": contributors,
            "degradation_warning": degradation_warning,
            "message": (
                "Potential sensor degradation detected. "
                "Inspection recommended."
                if degradation_warning
                else None
            ),
        }
```

## `skyguard_ml/explainability.py`

```python
from __future__ import annotations

from typing import Any


def build_reason_codes(
    anomaly_type: str,
    historical_score: float,
    temporal_score: float,
    multivariate_score: float,
    spatial_score: float | None,
    forecast_score: float | None,
    quality_flags: list[str],
) -> list[str]:
    codes: list[str] = []

    if anomaly_type != "UNKNOWN_ANOMALY":
        codes.append(anomaly_type)

    if historical_score >= 0.5:
        codes.append("HISTORICAL_DEVIATION")

    if temporal_score >= 0.5:
        codes.append("TEMPORAL_BEHAVIOUR")

    if multivariate_score >= 0.5:
        codes.append("MULTIVARIATE_INCONSISTENCY")

    if spatial_score is not None and spatial_score >= 0.5:
        codes.append("SPATIAL_DISAGREEMENT")

    if forecast_score is not None and forecast_score >= 0.5:
        codes.append("FORECAST_INCONSISTENCY")

    for flag in quality_flags:
        if flag not in codes:
            codes.append(flag)

    return codes


def build_explanation(
    status: str,
    anomaly_type: str,
    evidence: dict[str, Any],
    reason_codes: list[str],
    event_classification: str,
) -> str:
    statements: list[str] = []

    if status == "NORMAL":
        statements.append(
            "The observation is within the currently available "
            "historical, temporal, and multivariate evidence limits."
        )
    else:
        statements.append(
            f"The observation was classified as {status.lower()} "
            f"with anomaly type {anomaly_type}."
        )

    historical_score = evidence.get("historical_score", 0.0)
    temporal_score = evidence.get("temporal_score", 0.0)
    ml_score = evidence.get("ml_score", 0.0)

    if historical_score >= 0.5:
        statements.append(
            f"Historical deviation evidence was "
            f"{historical_score:.2f}."
        )

    if temporal_score >= 0.5:
        statements.append(
            f"Temporal-pattern evidence was "
            f"{temporal_score:.2f}."
        )

    if ml_score >= 0.5:
        statements.append(
            f"The selected ML model produced an anomaly score "
            f"of {ml_score:.2f}."
        )

    if evidence.get("spatial_score") is not None:
        statements.append(
            f"Spatial disagreement evidence was "
            f"{evidence['spatial_score']:.2f}."
        )

    if evidence.get("forecast_score") is not None:
        statements.append(
            f"Forecast inconsistency evidence was "
            f"{evidence['forecast_score']:.2f}."
        )

    statements.append(
        f"Context classification: {event_classification}."
    )

    return " ".join(statements)


def recommended_action(
    status: str,
    event_classification: str,
    anomaly_type: str,
) -> str:
    if anomaly_type in {
        "MISSING_DATA",
        "COMMUNICATION_FAILURE",
        "DUPLICATE_DATA",
        "TIMESTAMP_ERROR",
    }:
        return "Check station connectivity, timestamp handling, and data ingestion."

    if event_classification == "LIKELY_WEATHER_EVENT":
        return "Retain the observation and compare it with broader weather conditions."

    if event_classification == "LIKELY_SENSOR_ANOMALY":
        return "Inspect the sensor and compare against a calibrated reference."

    if status == "CRITICAL":
        return "Escalate for immediate station inspection."

    if status == "WARNING":
        return "Continue monitoring and inspect if the pattern persists."

    return "No immediate action required."
```

## `skyguard_ml/model_selection.py`

```python
from __future__ import annotations

from typing import Any


def calculate_selection_score(
    metrics: dict[str, float],
    weights: dict[str, float],
) -> float:
    return (
        weights.get("f1", 0.0) * metrics.get("f1", 0.0)
        + weights.get("recall", 0.0) * metrics.get("recall", 0.0)
        + weights.get("precision", 0.0)
        * metrics.get("precision", 0.0)
        - weights.get("false_positive_rate", 0.0)
        * metrics.get("false_positive_rate", 0.0)
        - weights.get("false_alarm_rate", 0.0)
        * min(metrics.get("false_alarms_per_station_day", 0.0), 1.0)
    )


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
```

## `skyguard_ml/evaluation.py`

```python
from __future__ import annotations

from typing import Any
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_detector(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()

    predictions = model.predict(x)
    scores = model.score_samples(x)

    inference_time = time.perf_counter() - start

    y_true = np.asarray(y).astype(int)
    y_pred = np.asarray(predictions).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    false_positive_rate = fp / max(fp + tn, 1)

    false_alarms_per_station_day = float(false_positive_rate)

    detection_latency = None
    anomaly_indices = np.where(y_true == 1)[0]

    if len(anomaly_indices):
        detected = anomaly_indices[y_pred[anomaly_indices] == 1]
        if len(detected):
            detection_latency = float(
                np.mean(
                    [
                        int(index - anomaly_indices[0])
                        for index in detected
                    ]
                )
            )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(false_positive_rate),
        "false_alarms_per_station_day": false_alarms_per_station_day,
        "detection_latency": detection_latency,
        "training_time": None,
        "inference_time": inference_time,
        "mean_anomaly_score": float(np.mean(scores)),
        "sample_count": len(y_true),
    }


def create_ablation_report(
    y_true: pd.Series,
    rules_predictions: np.ndarray,
    statistics_predictions: np.ndarray,
    ml_predictions: np.ndarray,
    spatial_predictions: np.ndarray,
    forecast_predictions: np.ndarray,
) -> pd.DataFrame:
    systems = {
        "Rules only": rules_predictions,
        "Rules + Statistics": statistics_predictions,
        "Rules + Statistics + Best ML Model": ml_predictions,
        "Rules + Statistics + ML + Spatial": spatial_predictions,
        "Rules + Statistics + ML + Spatial + Forecast": forecast_predictions,
    }

    rows: list[dict[str, Any]] = []

    for name, predictions in systems.items():
        rows.append(
            {
                "system": name,
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "false_positive_rate": float(
                    np.mean(
                        (np.asarray(predictions) == 1)
                        & (np.asarray(y_true) == 0)
                    )
                ),
                "false_alarms_per_day": float(
                    np.mean(np.asarray(predictions) == 1)
                ),
            }
        )

    return pd.DataFrame(rows)
```

## `skyguard_ml/training.py`

```python
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .baselines import BaselineStore
from .config import TrainingConfig
from .evaluation import evaluate_detector
from .feature_engineering import BASE_VARIABLES, FeatureBuilder
from .model_selection import select_best_model
from .models import build_candidate_models, create_model
from .preprocessing import chronological_split, safe_numeric_frame
from .schemas import observations_to_frame
from .synthetic_faults import generate_synthetic_faults
from .validation import validate_observations


def train_all(
    observations: list[dict[str, Any]] | pd.DataFrame,
    config: TrainingConfig | None = None,
    artifact_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = config or TrainingConfig()

    if artifact_dir is not None:
        config.artifact_dir = str(artifact_dir)

    output_dir = Path(config.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_frame = observations_to_frame(observations)

    validated_frame, validation_report = validate_observations(
        raw_frame,
    )

    usable = validated_frame[
        validated_frame["station_id"].notna()
        & validated_frame["timestamp"].notna()
    ].copy()

    if len(usable) < config.min_training_rows:
        raise ValueError(
            f"At least {config.min_training_rows} usable rows are required."
        )

    baseline = BaselineStore(
        minimum_samples=config.feature.minimum_baseline_samples,
    )
    baseline.fit(usable, BASE_VARIABLES)

    builder = FeatureBuilder(config.feature)
    clean_features = builder.fit_transform(
        usable,
        baseline=baseline,
    )

    clean_train, clean_validation, clean_test = chronological_split(
        clean_features,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    if clean_train.empty:
        raise ValueError("Chronological training split is empty.")

    fault_frame, fault_metadata = generate_synthetic_faults(
        usable,
        faults_per_type=config.synthetic_faults_per_type,
        random_state=config.random_state,
    )

    fault_features = builder.transform(fault_frame)

    fault_train, fault_validation, fault_test = chronological_split(
        fault_features,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    train_normal = clean_train.copy()

    x_train = safe_numeric_frame(
        train_normal,
        builder.feature_list,
    )

    model_results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}

    for model_name, parameter_list in build_candidate_models(
        random_state=config.random_state,
    ).items():
        best_model = None
        best_metrics = None
        best_params = None

        for params in parameter_list:
            try:
                started = time.perf_counter()
                candidate = create_model(
                    model_name,
                    params,
                    random_state=config.random_state,
                )
                candidate.fit(x_train)

                x_validation = safe_numeric_frame(
                    fault_validation,
                    builder.feature_list,
                )
                y_validation = fault_validation["target"].fillna(0)

                metrics = evaluate_detector(
                    candidate,
                    x_validation,
                    y_validation,
                )

                metrics["training_time"] = (
                    time.perf_counter() - started
                )

                if (
                    best_metrics is None
                    or metrics["f1"] > best_metrics["f1"]
                ):
                    best_model = candidate
                    best_metrics = metrics
                    best_params = params

            except Exception as exc:
                model_results.setdefault(
                    model_name,
                    {},
                ).setdefault(
                    "errors",
                    [],
                ).append(str(exc))

        if best_model is None or best_metrics is None:
            continue

        trained_models[model_name] = best_model

        x_test = safe_numeric_frame(
            fault_test,
            builder.feature_list,
        )
        y_test = fault_test["target"].fillna(0)

        test_metrics = evaluate_detector(
            best_model,
            x_test,
            y_test,
        )

        model_results[model_name] = {
            "hyperparameters": best_params,
            "validation_metrics": best_metrics,
            "test_metrics": test_metrics,
        }

    if not model_results:
        raise RuntimeError("No candidate anomaly detector trained successfully.")

    selection_weights = config.selection_weights.as_dict()

    selected_name, selected_result = select_best_model(
        model_results,
        selection_weights,
    )

    selected_model = trained_models[selected_name]

    bundle = {
        "model": selected_model,
        "model_name": selected_name,
        "feature_builder": builder,
        "baseline": baseline,
        "config": config,
    }

    joblib.dump(
        bundle,
        output_dir / "best_model.joblib",
    )

    joblib.dump(
        {
            "models": trained_models,
            "results": model_results,
        },
        output_dir / "candidate_models.joblib",
    )

    (output_dir / "feature_config.json").write_text(
        json.dumps(
            {
                "feature_list": builder.feature_list,
                "feature_config": vars(config.feature),
            },
            indent=2,
            default=str,
        )
    )

    (output_dir / "preprocessing_config.json").write_text(
        json.dumps(
            {
                "numeric_variables": BASE_VARIABLES,
                "chronological_split": {
                    "train_fraction": config.train_fraction,
                    "validation_fraction": config.validation_fraction,
                    "test_fraction": config.test_fraction,
                },
            },
            indent=2,
        )
    )

    (output_dir / "baseline_statistics.json").write_text(
        json.dumps(
            baseline.stats,
            indent=2,
            default=str,
        )
    )

    (output_dir / "fusion_weights.json").write_text(
        json.dumps(
            config.fusion_weights.as_dict(),
            indent=2,
        )
    )

    metadata = {
        "selected_model": selected_name,
        "model_version": "0.1.0",
        "hyperparameters": selected_result["hyperparameters"],
        "training_period": {
            "start": str(usable["timestamp"].min()),
            "end": str(usable["timestamp"].max()),
        },
        "validation_metrics": selected_result["validation_metrics"],
        "test_metrics": selected_result["test_metrics"],
        "feature_list": builder.feature_list,
        "selection_score": selected_result["selection_score"],
        "validation_report": {
            "row_count": validation_report.row_count,
            "valid_row_count": validation_report.valid_row_count,
            "invalid_row_count": validation_report.invalid_row_count,
            "summary": validation_report.summary,
        },
        "autoencoder": "NOT_USED",
        "confidence_status": "NOT_CALIBRATED",
    }

    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )

    report = {
        "models": model_results,
        "selected_model": selected_name,
        "synthetic_fault_count": len(fault_metadata),
        "feature_count": len(builder.feature_list),
        "validation_summary": validation_report.summary,
    }

    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )

    return report
```

## `skyguard_ml/inference.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .anomaly_types import infer_anomaly_type
from .config import TrainingConfig
from .evidence_fusion import fuse_evidence
from .event_context import classify_event_context
from .explainability import (
    build_explanation,
    build_reason_codes,
    recommended_action,
)
from .feature_engineering import BASE_VARIABLES
from .schemas import PredictionResult, normalize_observation
from .sensor_health import SensorHealthTracker
from .spatial_analysis import analyze_spatial_context
from .forecast_analysis import analyze_forecast_context
from .statistical_detectors import run_statistical_detectors
from .validation import validate_observations


class SkyGuard:
    def __init__(
        self,
        artifact_dir: str | Path = "skyguard_ml/artifacts",
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.bundle: dict[str, Any] | None = None
        self.health_tracker = SensorHealthTracker()

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    def load(self) -> "SkyGuard":
        path = self.artifact_dir / "best_model.joblib"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing trained artifact: {path}"
            )

        self.bundle = joblib.load(path)
        return self

    def train(
        self,
        observations: list[dict[str, Any]] | pd.DataFrame,
        config: TrainingConfig | None = None,
    ) -> dict[str, Any]:
        from .training import train_all

        report = train_all(
            observations,
            config=config,
            artifact_dir=self.artifact_dir,
        )
        self.load()
        return report

    def predict(
        self,
        observation: dict[str, Any],
        historical_context: list[dict[str, Any]] | pd.DataFrame,
        forecast_context: dict[str, Any] | None = None,
        nearby_station_context: list[dict[str, Any]]
        | pd.DataFrame
        | None = None,
    ) -> dict[str, Any]:
        if not self.loaded:
            self.load()

        assert self.bundle is not None

        current = normalize_observation(observation)

        if isinstance(historical_context, pd.DataFrame):
            history = historical_context.copy()
        else:
            history = pd.DataFrame(historical_context)

        combined = pd.concat(
            [
                history,
                pd.DataFrame([current]),
            ],
            ignore_index=True,
        )

        validated, validation_report = validate_observations(
            combined,
        )

        validated = validated.sort_values(
            ["station_id", "timestamp"],
            kind="mergesort",
        ).reset_index(drop=True)

        builder = self.bundle["feature_builder"]
        baseline = self.bundle["baseline"]
        model = self.bundle["model"]

        features = builder.transform(validated)

        current_station = current["station_id"]
        current_timestamp = current["timestamp"]

        matching = features[
            (features["station_id"] == current_station)
            & (features["timestamp"] == current_timestamp)
        ]

        if matching.empty:
            matching = features.tail(1)

        current_features = matching.iloc[-1]

        x_current = current_features.reindex(
            builder.feature_list,
        ).to_frame().T

        ml_score = float(
            model.score_samples(x_current)[0]
        )

        target_history = validated[
            validated["station_id"] == current_station
        ].copy()

        statistical = run_statistical_detectors(
            history=target_history,
            current_features=current_features,
            variables=BASE_VARIABLES,
            frozen_tolerance=builder.config.frozen_tolerance,
            frozen_duration_hours=builder.config.frozen_duration_hours,
        )

        historical_scores: list[float] = []
        baseline_sources: dict[str, str] = {}
        baseline_details: dict[str, Any] = {}

        for variable in BASE_VARIABLES:
            score, source, stats = baseline.score(
                variable=variable,
                value=current.get(variable),
                station_id=current_station,
                timestamp=current_timestamp,
            )
            historical_scores.append(score)
            baseline_sources[variable] = source
            baseline_details[variable] = stats

        historical_score = max(historical_scores or [0.0])
        temporal_score = float(
            statistical.get("temporal_score", 0.0)
        )

        multivariate_values = [
            abs(float(current_features.get(column, 0.0)))
            for column in [
                "temperature_humidity_deviation",
                "temperature_pressure_deviation",
                "humidity_pressure_deviation",
            ]
            if pd.notna(current_features.get(column))
        ]

        multivariate_score = min(
            1.0,
            (sum(multivariate_values) / len(multivariate_values))
            if multivariate_values
            else 0.0,
        )

        spatial = analyze_spatial_context(
            observation=current,
            nearby_context=nearby_station_context,
            variables=BASE_VARIABLES,
        )

        forecast = analyze_forecast_context(
            observation=current,
            forecast_context=forecast_context,
            variables=BASE_VARIABLES,
        )

        spatial_score = spatial.get("spatial_score")
        forecast_score = forecast.get("forecast_score")

        data_quality_score = 1.0 - validation_report.quality_score

        evidence_values = {
            "data_quality": data_quality_score,
            "statistical": historical_score,
            "temporal": temporal_score,
            "ml": ml_score,
            "multivariate": multivariate_score,
            "spatial": spatial_score,
            "forecast": forecast_score,
        }

        weights = self.bundle["config"].fusion_weights.as_dict()

        fused = fuse_evidence(
            evidence_values,
            weights,
        )

        anomaly_score = float(fused["score"])

        if anomaly_score >= 0.85:
            status = "CRITICAL"
        elif anomaly_score >= 0.65:
            status = "ANOMALY"
        elif anomaly_score >= 0.40:
            status = "WARNING"
        else:
            status = "NORMAL"

        anomaly_type = infer_anomaly_type(
            statistical=statistical,
            quality_flags=current.get("quality_flags", []),
            multivariate_score=multivariate_score,
        )

        event_classification = classify_event_context(
            anomaly_score=anomaly_score,
            historical_score=historical_score,
            temporal_score=temporal_score,
            multivariate_score=multivariate_score,
            spatial_score=spatial_score,
            forecast_score=forecast_score,
        )

        severity = self._severity(
            anomaly_score=anomaly_score,
            temporal_score=temporal_score,
            historical_score=historical_score,
            anomaly_type=anomaly_type,
            quality_score=data_quality_score,
        )

        health = self.health_tracker.update(
            station_id=str(current_station),
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            missing_score=statistical["missing"]["score"],
            drift_score=max(
                (
                    value["score"]
                    for value in statistical["drift"].values()
                ),
                default=0.0,
            ),
            spatial_disagreement=spatial.get(
                "sensor_disagreement_evidence"
            ),
            forecast_score=forecast_score,
        )

        evidence = {
            "ml_score": round(ml_score, 4),
            "historical_score": round(historical_score, 4),
            "temporal_score": round(temporal_score, 4),
            "multivariate_score": round(
                multivariate_score,
                4,
            ),
            "spatial_score": (
                None
                if spatial_score is None
                else round(spatial_score, 4)
            ),
            "forecast_score": (
                None
                if forecast_score is None
                else round(forecast_score, 4)
            ),
            "data_quality_score": round(
                data_quality_score,
                4,
            ),
            "baseline_sources": baseline_sources,
            "baseline_statistics": baseline_details,
            "statistical_detectors": statistical,
            "spatial": spatial,
            "forecast": forecast,
            "quality_flags": current.get("quality_flags", []),
            "used_sources": fused["used_sources"],
            "unavailable_sources": fused["unavailable_sources"],
        }

        reason_codes = build_reason_codes(
            anomaly_type=anomaly_type,
            historical_score=historical_score,
            temporal_score=temporal_score,
            multivariate_score=multivariate_score,
            spatial_score=spatial_score,
            forecast_score=forecast_score,
            quality_flags=current.get("quality_flags", []),
        )

        explanation = build_explanation(
            status=status,
            anomaly_type=anomaly_type,
            evidence=evidence,
            reason_codes=reason_codes,
            event_classification=event_classification,
        )

        action = recommended_action(
            status=status,
            event_classification=event_classification,
            anomaly_type=anomaly_type,
        )

        result = PredictionResult(
            status=status,
            anomaly_score=anomaly_score,
            selected_model=self.bundle["model_name"],
            anomaly_type=anomaly_type,
            severity=severity,
            event_classification=event_classification,
            evidence=evidence,
            sensor_health=health,
            reason_codes=reason_codes,
            explanation=explanation,
            recommended_action=action,
        )

        return result.to_dict()

    @staticmethod
    def _severity(
        anomaly_score: float,
        temporal_score: float,
        historical_score: float,
        anomaly_type: str,
        quality_score: float,
    ) -> str:
        score = (
            0.40 * anomaly_score
            + 0.20 * temporal_score
            + 0.20 * historical_score
            + 0.20 * quality_score
        )

        if anomaly_type in {
            "COMMUNICATION_FAILURE",
            "MISSING_DATA",
            "TIMESTAMP_ERROR",
        }:
            score += 0.10

        if score >= 0.85:
            return "CRITICAL"
        if score >= 0.65:
            return "HIGH"
        if score >= 0.40:
            return "MEDIUM"
        return "LOW"
```

## `skyguard_ml/__init__.py`

```python
from .config import (
    FeatureConfig,
    FusionWeights,
    SelectionWeights,
    TrainingConfig,
)
from .evaluation import create_ablation_report, evaluate_detector
from .inference import SkyGuard
from .model_selection import select_best_model
from .training import train_all


skyguard = SkyGuard()


def train(
    observations,
    config: TrainingConfig | None = None,
    artifact_dir: str = "skyguard_ml/artifacts",
):
    return skyguard.train(
        observations,
        config=config,
    )


def predict(
    observation,
    historical_context,
    forecast_context=None,
    nearby_station_context=None,
):
    return skyguard.predict(
        observation=observation,
        historical_context=historical_context,
        forecast_context=forecast_context,
        nearby_station_context=nearby_station_context,
    )


__all__ = [
    "SkyGuard",
    "skyguard",
    "train",
    "predict",
    "train_all",
    "evaluate_detector",
    "create_ablation_report",
    "select_best_model",
    "TrainingConfig",
    "FeatureConfig",
    "FusionWeights",
    "SelectionWeights",
]
```

## Basic Usage

```python
import pandas as pd

from skyguard_ml import SkyGuard, TrainingConfig


historical_data = pd.read_csv("historical_weather.csv")

historical_data["timestamp"] = pd.to_datetime(
    historical_data["timestamp"],
    utc=True,
)

config = TrainingConfig(
    artifact_dir="skyguard_ml/artifacts",
    min_training_rows=30,
    synthetic_faults_per_type=10,
)

engine = SkyGuard("skyguard_ml/artifacts")

training_report = engine.train(
    observations=historical_data,
    config=config,
)

print(training_report["selected_model"])
```

## Prediction

```python
observation = {
    "station_id": "ESP32_001",
    "timestamp": "2026-09-09T12:00:00Z",
    "temperature": 44.0,
    "humidity": 55.0,
    "pressure": 1008.3,
    "rainfall": 0.0,
    "wind_speed": 12.4,
    "wind_direction": 180.0,
    "latitude": 16.5,
    "longitude": 80.6,
}

historical_context = [
    {
        "station_id": "ESP32_001",
        "timestamp": "2026-09-09T11:00:00Z",
        "temperature": 35.2,
        "humidity": 64.0,
        "pressure": 1008.5,
        "rainfall": 0.0,
        "wind_speed": 11.8,
        "wind_direction": 175.0,
        "latitude": 16.5,
        "longitude": 80.6,
    },
    {
        "station_id": "ESP32_001",
        "timestamp": "2026-09-09T11:30:00Z",
        "temperature": 35.6,
        "humidity": 63.4,
        "pressure": 1008.4,
        "rainfall": 0.0,
        "wind_speed": 12.0,
        "wind_direction": 178.0,
        "latitude": 16.5,
        "longitude": 80.6,
    },
]

forecast_context = {
    "temperature": {
        "value": 35.0,
        "min": 32.0,
        "max": 37.0,
    },
    "humidity": {
        "value": 62.0,
        "min": 50.0,
        "max": 75.0,
    },
    "pressure": {
        "value": 1008.0,
        "min": 1004.0,
        "max": 1012.0,
    },
}

nearby_station_context = [
    {
        "station_id": "NEARBY_001",
        "timestamp": "2026-09-09T12:00:00Z",
        "temperature": 34.8,
        "humidity": 63.0,
        "pressure": 1008.0,
        "rainfall": 0.0,
        "wind_speed": 12.0,
        "wind_direction": 180.0,
        "latitude": 16.6,
        "longitude": 80.7,
    },
]

result = engine.predict(
    observation=observation,
    historical_context=historical_context,
    forecast_context=forecast_context,
    nearby_station_context=nearby_station_context,
)

print(result)
```

Example result shape:

```python
{
    "status": "ANOMALY",
    "anomaly_score": 0.87,
    "selected_model": "IsolationForest",
    "anomaly_type": "TEMPERATURE_SPIKE",
    "severity": "HIGH",
    "event_classification": "LIKELY_SENSOR_ANOMALY",
    "evidence": {
        "ml_score": 0.87,
        "historical_score": 0.91,
        "temporal_score": 0.88,
        "multivariate_score": 0.73,
        "spatial_score": 0.82,
        "forecast_score": 0.90,
        "data_quality_score": 0.0,
        "used_sources": [
            "data_quality",
            "statistical",
            "temporal",
            "ml",
            "multivariate",
            "spatial",
            "forecast",
        ],
        "unavailable_sources": [],
    },
    "sensor_health": {
        "score": 72.0,
        "status": "WARNING",
        "trend": "STABLE",
        "contributors": [
            "Repeated or strong anomaly evidence",
            "Disagreement with nearby stations",
            "Repeated forecast disagreement",
        ],
        "degradation_warning": False,
        "message": None,
    },
    "reason_codes": [
        "TEMPERATURE_SPIKE",
        "HISTORICAL_DEVIATION",
        "TEMPORAL_BEHAVIOUR",
        "MULTIVARIATE_INCONSISTENCY",
        "SPATIAL_DISAGREEMENT",
        "FORECAST_INCONSISTENCY",
    ],
    "explanation": "...",
    "recommended_action": "Inspect the sensor and compare against a calibrated reference.",
    "confidence_status": "NOT_CALIBRATED",
}
```

## Tests

### `tests/test_validation.py`

```python
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
```

### `tests/test_features.py`

```python
import pandas as pd

from skyguard_ml.baselines import BaselineStore
from skyguard_ml.config import FeatureConfig
from skyguard_ml.feature_engineering import (
    BASE_VARIABLES,
    FeatureBuilder,
)


def make_data():
    timestamps = pd.date_range(
        "2026-01-01",
        periods=20,
        freq="30min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "station_id": ["S1"] * 20,
            "timestamp": timestamps,
            "temperature": range(20),
            "humidity": [60.0] * 20,
            "pressure": [1008.0] * 20,
            "rainfall": [0.0] * 20,
            "wind_speed": [5.0] * 20,
            "wind_direction": [180.0] * 20,
            "latitude": [16.5] * 20,
            "longitude": [80.6] * 20,
        }
    )


def test_lag_does_not_use_future_value():
    frame = make_data()

    baseline = BaselineStore(minimum_samples=2)
    baseline.fit(frame, BASE_VARIABLES)

    builder = FeatureBuilder(
        FeatureConfig(lags=(1,), rolling_windows=(3,))
    )

    features = builder.fit_transform(frame, baseline)

    first_row = features.iloc[0]

    assert pd.isna(first_row["temperature_lag_1"])
    assert pd.isna(first_row["temperature_delta"])
```

### `tests/test_detectors.py`

```python
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
```

### `tests/test_end_to_end.py`

```python
import pandas as pd

from skyguard_ml import SkyGuard, TrainingConfig


def make_training_data(rows=80):
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="30min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "station_id": ["S1"] * rows,
            "timestamp": timestamps,
            "temperature": [
                30.0 + (index % 8) * 0.2
                for index in range(rows)
            ],
            "humidity": [60.0] * rows,
            "pressure": [1008.0] * rows,
            "rainfall": [0.0] * rows,
            "wind_speed": [5.0] * rows,
            "wind_direction": [180.0] * rows,
            "latitude": [16.5] * rows,
            "longitude": [80.6] * rows,
        }
    )


def test_training_and_prediction(tmp_path):
    data = make_training_data()
    artifact_dir = tmp_path / "artifacts"

    engine = SkyGuard(artifact_dir)

    report = engine.train(
        data,
        TrainingConfig(
            min_training_rows=30,
            synthetic_faults_per_type=2,
        ),
    )

    assert report["selected_model"]

    observation = data.iloc[-1].to_dict()
    history = data.iloc[:-1].to_dict(orient="records")

    result = engine.predict(
        observation=observation,
        historical_context=history,
    )

    assert "status" in result
    assert "anomaly_score" in result
    assert "sensor_health" in result
    assert "reason_codes" in result
```

## Important Notes

This implementation deliberately returns:

```text
confidence_status = NOT_CALIBRATED
```

because anomaly scores from Isolation Forest, LOF, One-Class SVM, and robust covariance are not automatically calibrated probabilities.

The autoencoder is not trained by default:

```text
AUTOENCODER = NOT USED
```

This is intentional. It should only be added when the dataset contains enough sequential normal data and the reconstruction threshold can be validated properly.

The saved artifacts are:

```text
skyguard_ml/artifacts/
├── best_model.joblib
├── candidate_models.joblib
├── feature_config.json
├── preprocessing_config.json
├── baseline_statistics.json
├── fusion_weights.json
├── model_metadata.json
└── evaluation_report.json
```

Run the tests with:

```bash
pytest
```

Train the model with:

```python
from skyguard_ml import SkyGuard, TrainingConfig

engine = SkyGuard("skyguard_ml/artifacts")

engine.train(
    observations,
    config=TrainingConfig(
        artifact_dir="skyguard_ml/artifacts",
        min_training_rows=30,
        synthetic_faults_per_type=10,
    ),
)
```

Then use the trained engine:

```python
result = engine.predict(
    observation,
    historical_context,
    forecast_context=None,
    nearby_station_context=None,
)
```
