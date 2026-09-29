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
