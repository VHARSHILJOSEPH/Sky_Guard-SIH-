from __future__ import annotations

"""Skyguard ML Models — Phase 8 candidate models with calibrated score normalization."""

from typing import Any

import numpy as np
import pandas as pd

from .isolation_forest import build_isolation_forest
from .lof import build_lof, build_lof_default, build_lof_sensitive, build_lof_strict
from .one_class_svm import build_one_class_svm
from .robust_covariance import build_robust_covariance
from .rules import check_physical_bounds, PHYSICAL_BOUNDS


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
        self.scale = 1.0

    def fit(self, x: pd.DataFrame) -> "DetectorModel":
        self.estimator.fit(x)

        raw = self._raw_score(x)
        finite = raw[np.isfinite(raw)]

        if len(finite) > 2:
            p75 = float(np.percentile(finite, 75))
            p25 = float(np.percentile(finite, 25))
            iqr = max(p75 - p25, 1e-4)
            # Scale factor so that raw scores around 0 are smoothly mapped around 0.5
            self.scale = float(2.0 / iqr)

        return self

    def _raw_score(self, x: pd.DataFrame) -> np.ndarray:
        """Get raw decision_function scores from the estimator.

        For Pipeline estimators, Pipeline.decision_function() correctly
        applies all preprocessing steps (imputer, scaler) before
        calling the final estimator's decision_function.
        """
        model = self.estimator

        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(x))

        # Fallback for non-pipeline models
        final_model = model[-1]
        return np.asarray(final_model.decision_function(x))

    def score_samples(self, x: pd.DataFrame) -> np.ndarray:
        """Return anomaly scores in [0, 1] where 1 = most anomalous.

        In scikit-learn anomaly detectors:
          - decision_function(x) > 0: INLIER (normal)
          - decision_function(x) = 0: DECISION BOUNDARY
          - decision_function(x) < 0: OUTLIER (anomaly)

        We use a logistic sigmoid centered at 0:
          score = 1.0 / (1.0 + exp(scale * raw))
          - raw >> 0 (normal) -> score near 0.0
          - raw == 0 (boundary) -> score = 0.50
          - raw << 0 (anomaly) -> score near 1.0
        """
        raw = self._raw_score(x)
        scaled = np.clip(self.scale * raw, -30.0, 30.0)
        anomaly_score = 1.0 / (1.0 + np.exp(scaled))
        return np.clip(anomaly_score, 0.0, 1.0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        raw = self._raw_score(x)
        return np.where(raw < 0, 1, 0)


def build_candidate_models(
    random_state: int = 42,
    max_rows: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Build the candidate model hyperparameter grid.

    Phase 8: Evaluates LOF, IsolationForest, OneClassSVM, RobustCovariance
    with calibrated contamination and outlier parameters.
    """
    candidates: dict[str, list[dict[str, Any]]] = {
        "IsolationForest": [
            {
                "n_estimators": 200,
                "max_samples": "auto",
                "contamination": 0.02,
                "max_features": 1.0,
            },
            {
                "n_estimators": 300,
                "max_samples": "auto",
                "contamination": 0.05,
                "max_features": 0.8,
            },
        ],
        "LOF": [
            {
                "n_neighbors": 20,
                "contamination": 0.02,
                "metric": "minkowski",
            },
            {
                "n_neighbors": 15,
                "contamination": 0.03,
                "metric": "minkowski",
            },
        ],
        "OneClassSVM": [
            {
                "kernel": "rbf",
                "nu": 0.03,
                "gamma": "scale",
            },
            {
                "kernel": "rbf",
                "nu": 0.05,
                "gamma": "scale",
            },
        ],
        "RobustCovariance": [
            {
                "contamination": 0.02,
                "support_fraction": None,
            },
            {
                "contamination": 0.05,
                "support_fraction": 0.9,
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
