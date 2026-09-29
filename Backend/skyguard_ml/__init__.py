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
