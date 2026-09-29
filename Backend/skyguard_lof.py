from __future__ import annotations

import copy
import math
import numbers
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


# ============================================================
# Configuration
# ============================================================

BME280_PARAMETERS = (
    "temperature",
    "humidity",
    "pressure",
)

FAULT_TYPES = (
    "TEMPERATURE_SPIKE",
    "TEMPERATURE_DROP",
    "TEMPERATURE_DRIFT",
    "HUMIDITY_ANOMALY",
    "PRESSURE_ANOMALY",
    "FROZEN_SENSOR",
    "MISSING_DATA",
)

FEATURE_SCHEMA_VERSION = "BME280_FEATURES_v1"
MODEL_VERSION = "LOF_BME280_v1.0"


@dataclass
class SkyGuardLOFConfig:
    model_name: str = "LOF"
    model_version: str = MODEL_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    n_neighbors: int = 20
    contamination: str | float = "auto"
    n_jobs: int = -1

    rolling_window_hours: int = 24
    min_history_length: int = 24

    threshold_method: str = "validation_f1"
    random_state: int = 42

    # Expected units for BME280 input.
    temperature_unit: str = "celsius"
    humidity_unit: str = "percent_rh"
    pressure_unit: str = "hpa"


def make_feature_names() -> list[str]:
    names: list[str] = []

    suffixes = (
        "current",
        "lag1",
        "diff1",
        "rolling_mean_24h",
        "rolling_std_24h",
        "rolling_min_24h",
        "rolling_max_24h",
    )

    for parameter in BME280_PARAMETERS:
        for suffix in suffixes:
            names.append(f"{parameter}_{suffix}")

    return names


FEATURE_NAMES = make_feature_names()


# ============================================================
# Exceptions
# ============================================================

class SkyGuardValidationError(ValueError):
    """Raised when an input violates the model data contract."""


class ModelNotFittedError(RuntimeError):
    """Raised when inference is requested before fitting."""


# ============================================================
# Validation and timestamps
# ============================================================

def parse_timestamp(value: Any, *, required: bool = True) -> pd.Timestamp:
    if value is None:
        if required:
            raise SkyGuardValidationError("timestamp is required")
        return pd.Timestamp(datetime.now(timezone.utc))

    try:
        parsed = pd.Timestamp(value)
    except Exception as exc:
        raise SkyGuardValidationError(
            f"Invalid timestamp: {value!r}"
        ) from exc

    if pd.isna(parsed):
        raise SkyGuardValidationError("timestamp cannot be NaN")

    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")

    return parsed


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
    except Exception:
        return False

    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _is_numeric(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, (bool, np.bool_))
    )


def _validate_units(observation: Mapping[str, Any]) -> None:
    units = observation.get("units")

    if units is None:
        return

    if not isinstance(units, Mapping):
        raise SkyGuardValidationError("units must be a mapping")

    expected = {
        "temperature": "celsius",
        "humidity": "percent_rh",
        "pressure": "hpa",
    }

    for parameter, expected_unit in expected.items():
        supplied = units.get(parameter)

        if supplied is not None and supplied != expected_unit:
            raise SkyGuardValidationError(
                f"{parameter} must use unit {expected_unit!r}; "
                f"received {supplied!r}"
            )


def validate_observation(
    observation: Mapping[str, Any],
    *,
    require_timestamp: bool = True,
    allow_missing_measurements: bool = True,
) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise SkyGuardValidationError("observation must be a mapping")

    _validate_units(observation)

    timestamp = parse_timestamp(
        observation.get("timestamp"),
        required=require_timestamp,
    )

    missing_fields: list[str] = []
    invalid_fields: list[str] = []
    values: dict[str, Optional[float]] = {}

    for parameter in BME280_PARAMETERS:
        if parameter not in observation:
            missing_fields.append(parameter)
            values[parameter] = None
            continue

        value = observation.get(parameter)

        if _is_missing(value):
            missing_fields.append(parameter)
            values[parameter] = None
            continue

        if not _is_numeric(value):
            invalid_fields.append(parameter)
            values[parameter] = None
            continue

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            invalid_fields.append(parameter)
            values[parameter] = None
            continue

        # Basic unit/physical checks. More detailed physical QC belongs
        # to the outer SkyGuard evidence-fusion system.
        if parameter == "humidity" and not 0.0 <= numeric_value <= 100.0:
            invalid_fields.append(parameter)
            values[parameter] = None
            continue

        if parameter == "pressure" and numeric_value <= 0.0:
            invalid_fields.append(parameter)
            values[parameter] = None
            continue

        values[parameter] = numeric_value

    if invalid_fields:
        raise SkyGuardValidationError(
            f"Invalid BME280 fields: {invalid_fields}"
        )

    if missing_fields and not allow_missing_measurements:
        raise SkyGuardValidationError(
            f"Missing BME280 fields: {missing_fields}"
        )

    return {
        "timestamp": timestamp,
        "values": values,
        "missing_fields": missing_fields,
    }


def _normalise_observation(
    observation: Mapping[str, Any],
    *,
    station_id: Optional[str] = None,
    require_timestamp: bool = True,
) -> dict[str, Any]:
    validated = validate_observation(
        observation,
        require_timestamp=require_timestamp,
        allow_missing_measurements=True,
    )

    result = dict(observation)
    result["timestamp"] = validated["timestamp"]
    result["station_id"] = (
        station_id
        if station_id is not None
        else observation.get("station_id")
    )

    for parameter, value in validated["values"].items():
        result[parameter] = value

    return result


def _clean_history(
    history: Sequence[Mapping[str, Any]],
    station_id: str,
    current_timestamp: pd.Timestamp,
) -> list[dict[str, Any]]:
    if history is None:
        return []

    cleaned: list[dict[str, Any]] = []

    for item in history:
        if not isinstance(item, Mapping):
            raise SkyGuardValidationError(
                "Every history item must be a mapping"
            )

        item_station = item.get("station_id")

        if item_station is not None and item_station != station_id:
            raise SkyGuardValidationError(
                "History contains observations from another station"
            )

        normalised = _normalise_observation(
            item,
            station_id=station_id,
            require_timestamp=True,
        )

        item_timestamp = normalised["timestamp"]

        # Future records are deliberately excluded. This protects the
        # feature builder even if the backend sends a wider time range.
        if item_timestamp < current_timestamp:
            cleaned.append(normalised)

    cleaned.sort(key=lambda row: row["timestamp"])
    return cleaned


# ============================================================
# Causal feature engineering
# ============================================================

def _safe_stat(values: Sequence[Any], operation: str) -> float:
    numeric_values = [
        float(value)
        for value in values
        if value is not None
        and _is_numeric(value)
        and math.isfinite(float(value))
    ]

    if not numeric_values:
        return float("nan")

    array = np.asarray(numeric_values, dtype=float)

    if operation == "mean":
        return float(np.mean(array))
    if operation == "std":
        return float(np.std(array, ddof=0))
    if operation == "min":
        return float(np.min(array))
    if operation == "max":
        return float(np.max(array))

    raise ValueError(f"Unsupported statistic: {operation}")


def build_features(
    history: Sequence[Mapping[str, Any]],
    observation: Mapping[str, Any],
    *,
    station_id: Optional[str] = None,
    config: Optional[SkyGuardLOFConfig] = None,
) -> dict[str, float]:
    """
    Build one causal feature row.

    Only history observations earlier than the current observation are used.
    The rolling statistics include the current observation and previous
    observations within the configured time window.
    """
    config = config or SkyGuardLOFConfig()

    current = _normalise_observation(
        observation,
        station_id=station_id,
        require_timestamp=False,
    )

    resolved_station_id = (
        station_id
        or current.get("station_id")
        or observation.get("station_id")
    )

    if not resolved_station_id:
        raise SkyGuardValidationError("station_id is required")

    current_timestamp = current["timestamp"]

    previous = _clean_history(
        history,
        str(resolved_station_id),
        current_timestamp,
    )

    rows = previous + [current]

    if len(previous) < config.min_history_length:
        raise SkyGuardValidationError(
            "Insufficient temporal history for feature generation"
        )

    feature_row: dict[str, float] = {}

    rolling_start = (
        current_timestamp
        - pd.Timedelta(hours=config.rolling_window_hours)
    )

    rolling_rows = [
        row
        for row in rows
        if rolling_start <= row["timestamp"] <= current_timestamp
    ]

    for parameter in BME280_PARAMETERS:
        current_value = current.get(parameter)

        lag1_value = (
            previous[-1].get(parameter)
            if previous
            else None
        )

        if current_value is None:
            current_float = float("nan")
        else:
            current_float = float(current_value)

        if lag1_value is None:
            lag1_float = float("nan")
        else:
            lag1_float = float(lag1_value)

        if math.isfinite(current_float) and math.isfinite(lag1_float):
            diff1 = current_float - lag1_float
        else:
            diff1 = float("nan")

        rolling_values = [
            row.get(parameter)
            for row in rolling_rows
        ]

        feature_row[f"{parameter}_current"] = current_float
        feature_row[f"{parameter}_lag1"] = lag1_float
        feature_row[f"{parameter}_diff1"] = diff1
        feature_row[
            f"{parameter}_rolling_mean_24h"
        ] = _safe_stat(rolling_values, "mean")
        feature_row[
            f"{parameter}_rolling_std_24h"
        ] = _safe_stat(rolling_values, "std")
        feature_row[
            f"{parameter}_rolling_min_24h"
        ] = _safe_stat(rolling_values, "min")
        feature_row[
            f"{parameter}_rolling_max_24h"
        ] = _safe_stat(rolling_values, "max")

    return feature_row


@dataclass
class FeatureDataset:
    X: pd.DataFrame
    records: list[dict[str, Any]]
    y: Optional[np.ndarray] = None


def build_feature_dataset(
    observations: Sequence[Mapping[str, Any]],
    *,
    labels: Optional[Sequence[int]] = None,
    config: Optional[SkyGuardLOFConfig] = None,
) -> FeatureDataset:
    """
    Create causal features from a chronological or unordered observation list.

    Rows without enough prior station history are removed. The returned
    records are aligned with X and y.
    """
    config = config or SkyGuardLOFConfig()

    if not observations:
        raise SkyGuardValidationError("observations cannot be empty")

    if labels is not None and len(labels) != len(observations):
        raise SkyGuardValidationError(
            "labels must have the same length as observations"
        )

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}

    for index, raw_observation in enumerate(observations):
        normalised = _normalise_observation(
            raw_observation,
            station_id=raw_observation.get("station_id"),
            require_timestamp=True,
        )

        station_id = normalised.get("station_id")

        if not station_id:
            raise SkyGuardValidationError(
                "Every training observation requires station_id"
            )

        grouped.setdefault(str(station_id), []).append(
            (index, normalised)
        )

    feature_rows: list[dict[str, float]] = []
    kept_records: list[dict[str, Any]] = []
    kept_labels: list[int] = []

    for station_id, station_rows in grouped.items():
        station_rows.sort(key=lambda item: item[1]["timestamp"])

        for position, (original_index, current) in enumerate(station_rows):
            if position < config.min_history_length:
                continue

            history = [
                row
                for _, row in station_rows[:position]
            ]

            row = build_features(
                history,
                current,
                station_id=station_id,
                config=config,
            )

            feature_rows.append(row)
            kept_records.append(current)

            if labels is not None:
                kept_labels.append(int(labels[original_index]))
            elif "is_anomaly" in current:
                kept_labels.append(int(current["is_anomaly"]))

    if not feature_rows:
        raise SkyGuardValidationError(
            "No feature rows were created. More history is required."
        )

    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)

    y: Optional[np.ndarray] = None

    if labels is not None or kept_labels:
        y = np.asarray(kept_labels, dtype=int)

    return FeatureDataset(
        X=X,
        records=kept_records,
        y=y,
    )


def chronological_feature_split(
    observations: Sequence[Mapping[str, Any]],
    *,
    config: Optional[SkyGuardLOFConfig] = None,
) -> tuple[FeatureDataset, FeatureDataset, FeatureDataset]:
    """
    Build all causal features first, then split chronologically.

    This preserves history across the train/validation/test boundaries.
    """
    dataset = build_feature_dataset(
        observations,
        config=config,
    )

    n_rows = len(dataset.X)
    train_end = int(n_rows * 0.60)
    validation_end = int(n_rows * 0.80)

    if train_end == 0 or validation_end <= train_end:
        raise SkyGuardValidationError(
            "Not enough feature rows for a 60/20/20 split"
        )

    return (
        FeatureDataset(
            X=dataset.X.iloc[:train_end].reset_index(drop=True),
            records=dataset.records[:train_end],
        ),
        FeatureDataset(
            X=dataset.X.iloc[train_end:validation_end].reset_index(
                drop=True
            ),
            records=dataset.records[train_end:validation_end],
        ),
        FeatureDataset(
            X=dataset.X.iloc[validation_end:].reset_index(drop=True),
            records=dataset.records[validation_end:],
        ),
    )


# ============================================================
# Synthetic fault generation
# ============================================================

def _copy_with_metadata(
    observation: Mapping[str, Any],
    *,
    is_anomaly: int,
    fault_type: str,
    affected_parameter: str,
    fault_id: Optional[str],
    fault_start_time: Optional[Any],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(observation))
    result["is_anomaly"] = int(is_anomaly)
    result["fault_type"] = fault_type
    result["affected_parameter"] = affected_parameter

    if fault_id is not None:
        result["fault_id"] = fault_id

    if fault_start_time is not None:
        result["fault_start_time"] = parse_timestamp(
            fault_start_time,
            required=True,
        ).isoformat()

    return result


def inject_synthetic_fault(
    observations: Sequence[Mapping[str, Any]],
    *,
    fault_type: str,
    start_index: int,
    duration: int = 1,
    parameter: Optional[str] = None,
    magnitude: Optional[float] = None,
    fault_id: str = "synthetic_fault_1",
) -> list[dict[str, Any]]:
    """
    Inject one synthetic fault for benchmarking only.

    Synthetic records must never be included in LOF training.
    """
    if fault_type not in FAULT_TYPES:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    if not observations:
        raise ValueError("observations cannot be empty")

    if start_index < 0 or start_index >= len(observations):
        raise IndexError("start_index is outside the observations")

    if duration <= 0:
        raise ValueError("duration must be positive")

    result = [
        _copy_with_metadata(
            observation,
            is_anomaly=0,
            fault_type="NONE",
            affected_parameter="none",
            fault_id=None,
            fault_start_time=None,
        )
        for observation in observations
    ]

    start = start_index
    end = min(start_index + duration, len(result))

    start_timestamp = result[start].get("timestamp")

    if fault_type == "TEMPERATURE_SPIKE":
        parameter = parameter or "temperature"
        magnitude = 8.0 if magnitude is None else magnitude

        for index in range(start, end):
            result[index][parameter] = (
                float(result[index][parameter]) + abs(magnitude)
            )

    elif fault_type == "TEMPERATURE_DROP":
        parameter = parameter or "temperature"
        magnitude = 8.0 if magnitude is None else magnitude

        for index in range(start, end):
            result[index][parameter] = (
                float(result[index][parameter]) - abs(magnitude)
            )

    elif fault_type == "TEMPERATURE_DRIFT":
        parameter = parameter or "temperature"
        magnitude = 0.5 if magnitude is None else magnitude

        for offset, index in enumerate(range(start, end), start=1):
            result[index][parameter] = (
                float(result[index][parameter]) + magnitude * offset
            )

    elif fault_type == "HUMIDITY_ANOMALY":
        parameter = parameter or "humidity"
        magnitude = 25.0 if magnitude is None else magnitude

        for index in range(start, end):
            value = float(result[index][parameter])
            result[index][parameter] = float(
                np.clip(value + magnitude, 0.0, 100.0)
            )

    elif fault_type == "PRESSURE_ANOMALY":
        parameter = parameter or "pressure"
        magnitude = 15.0 if magnitude is None else magnitude

        for index in range(start, end):
            result[index][parameter] = (
                float(result[index][parameter]) + magnitude
            )

    elif fault_type == "FROZEN_SENSOR":
        parameter = parameter or "all"

        freeze_source = start - 1

        if freeze_source < 0:
            raise ValueError(
                "FROZEN_SENSOR requires at least one previous observation"
            )

        if parameter == "all":
            parameters = BME280_PARAMETERS
        else:
            parameters = (parameter,)

        for index in range(start, end):
            for field in parameters:
                result[index][field] = result[freeze_source][field]

    elif fault_type == "MISSING_DATA":
        parameter = parameter or "all"

        if parameter == "all":
            parameters = BME280_PARAMETERS
        else:
            parameters = (parameter,)

        for index in range(start, end):
            for field in parameters:
                result[index][field] = None

    affected_parameter = parameter or "all"

    for index in range(start, end):
        result[index]["is_anomaly"] = 1
        result[index]["fault_type"] = fault_type
        result[index]["affected_parameter"] = affected_parameter
        result[index]["fault_id"] = fault_id
        result[index]["fault_start_time"] = parse_timestamp(
            start_timestamp,
            required=True,
        ).isoformat()

    return result


# ============================================================
# Model
# ============================================================

class SkyGuardLOF:
    """
    Production LOF anomaly detector for ESP32 + BME280 telemetry.

    The model answers only whether telemetry is statistically anomalous.
    It does not classify the cause as a sensor failure.
    """

    def __init__(
        self,
        config: Optional[SkyGuardLOFConfig] = None,
    ) -> None:
        self.config = config or SkyGuardLOFConfig()

        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[RobustScaler] = None
        self.lof: Optional[LocalOutlierFactor] = None

        self.threshold: Optional[float] = None

        self.score_low: Optional[float] = None
        self.score_high: Optional[float] = None

        self.training_date: Optional[str] = None
        self.training_observation_count: Optional[int] = None
        self.training_feature_row_count: Optional[int] = None
        self.training_period_start: Optional[str] = None
        self.training_period_end: Optional[str] = None
        self.dataset_version: Optional[str] = None
        self.training_time_seconds: Optional[float] = None
        self.calibration_metrics: Optional[dict[str, Any]] = None

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def _ensure_fitted(self) -> None:
        if (
            self.imputer is None
            or self.scaler is None
            or self.lof is None
        ):
            raise ModelNotFittedError("The LOF model has not been fitted")

    def _as_feature_matrix(
        self,
        X: Any,
    ) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [
                name
                for name in FEATURE_NAMES
                if name not in X.columns
            ]

            if missing:
                raise SkyGuardValidationError(
                    f"Feature columns missing: {missing}"
                )

            matrix = X[FEATURE_NAMES].to_numpy(dtype=float)

        else:
            matrix = np.asarray(X, dtype=float)

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise SkyGuardValidationError(
                "Feature input must be a two-dimensional matrix"
            )

        if matrix.shape[1] != len(FEATURE_NAMES):
            raise SkyGuardValidationError(
                f"Expected {len(FEATURE_NAMES)} features, "
                f"received {matrix.shape[1]}"
            )

        return matrix

    def _prepare_scaled_features(self, X: Any) -> np.ndarray:
        self._ensure_fitted()

        matrix = self._as_feature_matrix(X)

        imputed = self.imputer.transform(matrix)
        scaled = self.scaler.transform(imputed)

        return scaled

    def _raw_scores_from_features(self, X: Any) -> np.ndarray:
        scaled = self._prepare_scaled_features(X)

        # decision_function: higher means more normal.
        # Negating it creates a score where higher means more anomalous.
        return -self.lof.decision_function(scaled)

    def fit(
        self,
        X_train: Sequence[Mapping[str, Any]] | pd.DataFrame | np.ndarray,
        *,
        dataset_version: str = "unknown",
    ) -> "SkyGuardLOF":
        """
        Fit on clean normal BME280 observations only.

        X_train may be:
        - raw observations with station_id and timestamp
        - a DataFrame containing FEATURE_NAMES
        - a numeric feature matrix
        """
        started = time.perf_counter()

        raw_observations: Optional[Sequence[Mapping[str, Any]]] = None

        if isinstance(X_train, (pd.DataFrame, np.ndarray)):
            X_features = self._as_feature_matrix(X_train)
        else:
            raw_observations = list(X_train)

            for observation in raw_observations:
                if int(observation.get("is_anomaly", 0)) == 1:
                    raise SkyGuardValidationError(
                        "Synthetic or labelled anomalies cannot enter training"
                    )

            feature_dataset = build_feature_dataset(
                raw_observations,
                config=self.config,
            )

            X_features = feature_dataset.X.to_numpy(dtype=float)

        if len(X_features) <= self.config.n_neighbors:
            raise SkyGuardValidationError(
                "Training requires more rows than n_neighbors"
            )

        self.imputer = SimpleImputer(strategy="median")
        X_imputed = self.imputer.fit_transform(X_features)

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)

        self.lof = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            contamination=self.config.contamination,
            novelty=True,
            n_jobs=self.config.n_jobs,
        )

        self.lof.fit(X_scaled)

        # Training scores are used only for score normalization.
        training_raw_scores = -self.lof.decision_function(X_scaled)

        self.score_low = float(
            np.nanpercentile(training_raw_scores, 1.0)
        )
        self.score_high = float(
            np.nanpercentile(training_raw_scores, 99.0)
        )

        if self.score_high <= self.score_low:
            self.score_high = self.score_low + 1e-12

        self.training_date = datetime.now(timezone.utc).isoformat()
        self.training_observation_count = (
            len(raw_observations)
            if raw_observations is not None
            else len(X_features)
        )
        self.training_feature_row_count = len(X_features)
        self.dataset_version = dataset_version
        self.training_time_seconds = time.perf_counter() - started

        if raw_observations:
            timestamps = [
                parse_timestamp(item["timestamp"])
                for item in raw_observations
            ]

            self.training_period_start = min(timestamps).isoformat()
            self.training_period_end = max(timestamps).isoformat()

        return self

    def _normalise_score(self, raw_score: float) -> float:
        if self.score_low is None or self.score_high is None:
            return float("nan")

        value = (
            (raw_score - self.score_low)
            / (self.score_high - self.score_low)
        )

        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _classification_metrics(
        y_true: Sequence[int],
        y_pred: Sequence[int],
        raw_scores: Optional[Sequence[float]] = None,
    ) -> dict[str, Any]:
        y_true_array = np.asarray(y_true, dtype=int)
        y_pred_array = np.asarray(y_pred, dtype=int)

        tn, fp, fn, tp = confusion_matrix(
            y_true_array,
            y_pred_array,
            labels=[0, 1],
        ).ravel()

        denominator_specificity = tn + fp
        denominator_fpr = fp + tn
        denominator_fnr = fn + tp

        metrics: dict[str, Any] = {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": float(
                accuracy_score(y_true_array, y_pred_array)
            ),
            "precision": float(
                precision_score(
                    y_true_array,
                    y_pred_array,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_true_array,
                    y_pred_array,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_true_array,
                    y_pred_array,
                    zero_division=0,
                )
            ),
            "specificity": (
                float(tn / denominator_specificity)
                if denominator_specificity
                else None
            ),
            "fpr": (
                float(fp / denominator_fpr)
                if denominator_fpr
                else None
            ),
            "fnr": (
                float(fn / denominator_fnr)
                if denominator_fnr
                else None
            ),
        }

        if raw_scores is not None:
            scores = np.asarray(raw_scores, dtype=float)

            if len(np.unique(y_true_array)) >= 2:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true_array, scores)
                )
                metrics["pr_auc"] = float(
                    average_precision_score(y_true_array, scores)
                )
            else:
                metrics["roc_auc"] = None
                metrics["pr_auc"] = None

        return metrics

    def calibrate_threshold(
        self,
        X_validation: pd.DataFrame | np.ndarray,
        y_validation: Sequence[int],
    ) -> dict[str, Any]:
        """
        Select the frozen production threshold using validation labels.

        The test set must not be used here.
        """
        self._ensure_fitted()

        y = np.asarray(y_validation, dtype=int)

        if len(y) == 0:
            raise SkyGuardValidationError(
                "Validation labels cannot be empty"
            )

        matrix = self._as_feature_matrix(X_validation)

        if len(matrix) != len(y):
            raise SkyGuardValidationError(
                "Validation features and labels must have equal lengths"
            )

        raw_scores = self._raw_scores_from_features(matrix)

        candidates = np.unique(raw_scores)

        best_threshold: Optional[float] = None
        best_metrics: Optional[dict[str, Any]] = None

        for candidate in candidates:
            predictions = (raw_scores >= candidate).astype(int)

            metrics = self._classification_metrics(
                y,
                predictions,
                raw_scores,
            )

            if best_metrics is None:
                best_threshold = float(candidate)
                best_metrics = metrics
                continue

            current_key = (
                metrics["f1"],
                metrics["precision"],
                metrics["recall"],
            )

            best_key = (
                best_metrics["f1"],
                best_metrics["precision"],
                best_metrics["recall"],
            )

            if current_key > best_key:
                best_threshold = float(candidate)
                best_metrics = metrics

        if best_threshold is None or best_metrics is None:
            raise SkyGuardValidationError(
                "Unable to calibrate threshold"
            )

        self.threshold = best_threshold
        self.calibration_metrics = {
            "method": self.config.threshold_method,
            "threshold": self.threshold,
            **best_metrics,
        }

        return dict(self.calibration_metrics)

    def _feature_evidence(
        self,
        feature_row: Mapping[str, float],
    ) -> dict[str, float]:
        evidence: dict[str, float] = {}

        for parameter in BME280_PARAMETERS:
            for suffix in (
                "current",
                "lag1",
                "diff1",
                "rolling_mean_24h",
                "rolling_std_24h",
                "rolling_min_24h",
                "rolling_max_24h",
            ):
                name = f"{parameter}_{suffix}"
                value = feature_row.get(name)

                if value is not None and math.isfinite(float(value)):
                    evidence[name] = float(value)

            current = feature_row.get(f"{parameter}_current")
            mean = feature_row.get(
                f"{parameter}_rolling_mean_24h"
            )

            if (
                current is not None
                and mean is not None
                and math.isfinite(float(current))
                and math.isfinite(float(mean))
            ):
                evidence[
                    f"{parameter}_baseline_deviation"
                ] = float(current - mean)

        return evidence

    def score(
        self,
        observation: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
        *,
        station_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Return an LOF score without applying the anomaly threshold.
        """
        self._ensure_fitted()

        current = _normalise_observation(
            observation,
            station_id=station_id,
            require_timestamp=False,
        )

        resolved_station_id = (
            station_id
            or current.get("station_id")
            or observation.get("station_id")
        )

        if not resolved_station_id:
            raise SkyGuardValidationError("station_id is required")

        feature_row = build_features(
            history,
            current,
            station_id=str(resolved_station_id),
            config=self.config,
        )

        X = pd.DataFrame([feature_row], columns=FEATURE_NAMES)
        raw_score = float(self._raw_scores_from_features(X)[0])

        return {
            "raw_lof_score": raw_score,
            "normalized_lof_score": self._normalise_score(raw_score),
            "feature_evidence": self._feature_evidence(feature_row),
            "features_used": list(FEATURE_NAMES),
        }

    def predict(
        self,
        station_id: str,
        observation: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """
        Predict NORMAL or ANOMALY for one BME280 observation.
        """
        started = time.perf_counter()

        timestamp = observation.get("timestamp")

        try:
            current = _normalise_observation(
                observation,
                station_id=station_id,
                require_timestamp=False,
            )

            timestamp = current["timestamp"].isoformat()

            history_length = len(
                [
                    item
                    for item in history
                    if item.get("station_id") in (None, station_id)
                ]
            )

            if history_length < self.config.min_history_length:
                return {
                    "station_id": station_id,
                    "timestamp": timestamp,
                    "status": "WARMUP",
                    "model": self.config.model_name,
                    "model_version": self.config.model_version,
                    "ml_anomaly": False,
                    "reason": "Insufficient temporal history",
                    "required_history_length": (
                        self.config.min_history_length
                    ),
                    "available_history_length": history_length,
                }

            score_result = self.score(
                current,
                history,
                station_id=station_id,
            )

            raw_score = score_result["raw_lof_score"]

            if self.threshold is None:
                status = "UNCALIBRATED"
                ml_anomaly = False
                reason = (
                    "Threshold has not been calibrated using validation data"
                )
            else:
                status = "EVALUATED"
                ml_anomaly = bool(raw_score >= self.threshold)
                reason = None

            missing_fields = [
                parameter
                for parameter in BME280_PARAMETERS
                if current.get(parameter) is None
            ]

            result: dict[str, Any] = {
                "station_id": station_id,
                "timestamp": timestamp,
                "status": status,
                "model": self.config.model_name,
                "model_version": self.config.model_version,
                "feature_schema_version": (
                    self.config.feature_schema_version
                ),
                "ml_anomaly": ml_anomaly,
                "raw_lof_score": raw_score,
                "normalized_lof_score": (
                    score_result["normalized_lof_score"]
                ),
                "threshold": self.threshold,
                "features_used": score_result["features_used"],
                "feature_evidence": score_result["feature_evidence"],
                "missing_fields": missing_fields,
                "missing_data_imputed": bool(missing_fields),
                "inference_time_ms": (
                    (time.perf_counter() - started) * 1000.0
                ),
            }

            if reason is not None:
                result["reason"] = reason

            return result

        except SkyGuardValidationError as exc:
            return {
                "station_id": station_id,
                "timestamp": (
                    str(timestamp)
                    if timestamp is not None
                    else None
                ),
                "status": "INVALID_INPUT",
                "model": self.config.model_name,
                "model_version": self.config.model_version,
                "ml_anomaly": False,
                "reason": str(exc),
            }

    def predict_batch(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Batch inference.

        Each item must contain:
        {
            "station_id": "...",
            "observation": {...},
            "history": [...]
        }
        """
        return [
            self.predict(
                station_id=item["station_id"],
                observation=item["observation"],
                history=item["history"],
            )
            for item in items
        ]

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: Sequence[int],
        *,
        timestamps: Optional[Sequence[Any]] = None,
        fault_types: Optional[Sequence[str]] = None,
        fault_start_times: Optional[Sequence[Any]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate using a frozen threshold.

        Synthetic fault metrics are benchmark metrics only. They are not
        real-world sensor-failure accuracy.
        """
        self._ensure_fitted()

        if self.threshold is None:
            raise SkyGuardValidationError(
                "Calibrate the threshold before test evaluation"
            )

        started = time.perf_counter()

        matrix = self._as_feature_matrix(X_test)
        y = np.asarray(y_test, dtype=int)

        if len(matrix) != len(y):
            raise SkyGuardValidationError(
                "Test features and labels must have equal lengths"
            )

        raw_scores = self._raw_scores_from_features(matrix)
        predictions = (raw_scores >= self.threshold).astype(int)

        metrics = self._classification_metrics(
            y,
            predictions,
            raw_scores,
        )

        metrics["threshold"] = self.threshold
        metrics["benchmark_note"] = (
            "Fault labels represent synthetic benchmark ground truth "
            "and must not be reported as real-world accuracy."
        )

        if timestamps is not None:
            parsed_timestamps = [
                parse_timestamp(timestamp)
                for timestamp in timestamps
            ]

            if len(parsed_timestamps) != len(y):
                raise SkyGuardValidationError(
                    "timestamps and test labels must have equal lengths"
                )

            duration_days = (
                (
                    max(parsed_timestamps)
                    - min(parsed_timestamps)
                ).total_seconds()
                / 86400.0
            )

            duration_days = max(duration_days, 1.0 / 24.0)

            fp = metrics["fp"]
            metrics["false_alarms_per_day"] = float(
                fp / duration_days
            )

            metrics["inference_time_seconds"] = (
                time.perf_counter() - started
            )
            metrics["inference_time_ms_per_observation"] = (
                metrics["inference_time_seconds"]
                * 1000.0
                / max(len(y), 1)
            )
        else:
            metrics["false_alarms_per_day"] = None
            metrics["inference_time_seconds"] = (
                time.perf_counter() - started
            )
            metrics["inference_time_ms_per_observation"] = (
                metrics["inference_time_seconds"]
                * 1000.0
                / max(len(y), 1)
            )

        if fault_types is not None:
            if len(fault_types) != len(y):
                raise SkyGuardValidationError(
                    "fault_types and test labels must have equal lengths"
                )

            metrics["fault_wise"] = {}

            for fault_type in sorted(set(fault_types)):
                indexes = [
                    index
                    for index, value in enumerate(fault_types)
                    if value == fault_type
                ]

                fault_y = y[indexes]
                fault_predictions = predictions[indexes]
                fault_scores = raw_scores[indexes]

                fault_metrics = self._classification_metrics(
                    fault_y,
                    fault_predictions,
                    fault_scores,
                )

                fault_metrics["detection_latency_seconds"] = None

                if (
                    timestamps is not None
                    and fault_start_times is not None
                ):
                    fault_start_candidates = [
                        fault_start_times[index]
                        for index in indexes
                        if fault_start_times[index] is not None
                    ]

                    if fault_start_candidates:
                        start_time = min(
                            parse_timestamp(value)
                            for value in fault_start_candidates
                        )

                        detected_times = [
                            parse_timestamp(timestamps[index])
                            for index in indexes
                            if predictions[index] == 1
                            and parse_timestamp(
                                timestamps[index]
                            ) >= start_time
                        ]

                        if detected_times:
                            fault_metrics[
                                "detection_latency_seconds"
                            ] = float(
                                (
                                    min(detected_times) - start_time
                                ).total_seconds()
                            )

                metrics["fault_wise"][fault_type] = fault_metrics

        return metrics

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: str | Path) -> "SkyGuardLOF":
        loaded = joblib.load(path)

        if not isinstance(loaded, cls):
            raise TypeError(
                "The artifact does not contain a SkyGuardLOF model"
            )

        return loaded

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.config.model_name,
            "model_version": self.config.model_version,
            "feature_schema_version": (
                self.config.feature_schema_version
            ),
            "feature_names": list(FEATURE_NAMES),
            "bme280_parameters": list(BME280_PARAMETERS),
            "n_neighbors": self.config.n_neighbors,
            "contamination": self.config.contamination,
            "threshold": self.threshold,
            "threshold_method": self.config.threshold_method,
            "rolling_window_hours": (
                self.config.rolling_window_hours
            ),
            "min_history_length": (
                self.config.min_history_length
            ),
            "training_date": self.training_date,
            "training_observation_count": (
                self.training_observation_count
            ),
            "training_feature_row_count": (
                self.training_feature_row_count
            ),
            "training_period_start": self.training_period_start,
            "training_period_end": self.training_period_end,
            "dataset_version": self.dataset_version,
            "training_time_seconds": self.training_time_seconds,
            "calibration_metrics": self.calibration_metrics,
            "score_description": (
                "raw_lof_score = -LOF decision_function; "
                "higher values indicate more anomalous telemetry. "
                "It is not a probability."
            ),
            "normalization_description": (
                "normalized_lof_score is clipped to [0, 1] using "
                "the fitted model's 1st and 99th percentile training "
                "raw scores. It is not a probability."
            ),
        }


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_synthetic_records(
    model: SkyGuardLOF,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """
    Evaluate records generated by inject_synthetic_fault().
    """
    dataset = build_feature_dataset(
        records,
        labels=[
            int(record.get("is_anomaly", 0))
            for record in records
        ],
        config=model.config,
    )

    timestamps = [
        record["timestamp"]
        for record in dataset.records
    ]

    fault_types = [
        record.get("fault_type", "NONE")
        for record in dataset.records
    ]

    fault_start_times = [
        record.get("fault_start_time")
        for record in dataset.records
    ]

    return model.evaluate(
        dataset.X,
        dataset.y,
        timestamps=timestamps,
        fault_types=fault_types,
        fault_start_times=fault_start_times,
    )
