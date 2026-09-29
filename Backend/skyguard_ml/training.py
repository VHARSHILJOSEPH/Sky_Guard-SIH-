from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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
    clean_features["target"] = 0
    clean_features["fault_type"] = "NORMAL"

    # Chronological split on clean features
    clean_train, clean_validation, clean_test = chronological_split(
        clean_features,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    if clean_train.empty:
        raise ValueError("Chronological training split is empty.")

    # Generate synthetic faults on clean usable data
    fault_frame, fault_metadata = generate_synthetic_faults(
        usable,
        faults_per_type=config.synthetic_faults_per_type,
        random_state=config.random_state,
    )

    fault_features = builder.transform(fault_frame)
    if "target" not in fault_features.columns and "target" in fault_frame.columns:
        fault_features["target"] = fault_frame["target"].values
    if "fault_type" not in fault_features.columns and "fault_type" in fault_frame.columns:
        fault_features["fault_type"] = fault_frame["fault_type"].values

    # Chronological split on fault features
    _, fault_validation, fault_test = chronological_split(
        fault_features,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    # Clean-only training set (strictly normal data for unsupervised / one-class anomaly detection)
    train_normal = clean_train.copy()
    x_train = safe_numeric_frame(
        train_normal,
        builder.feature_list,
    )

    # Properly merged validation set: clean normal (target=0) + injected faults (target=1)
    val_frames = [clean_validation]
    if not fault_validation.empty:
        val_frames.append(fault_validation)
    val_combined = pd.concat(val_frames, ignore_index=True)

    x_val = safe_numeric_frame(val_combined, builder.feature_list)
    y_val = val_combined["target"].fillna(0).astype(int)

    # Properly merged test set: clean normal (target=0) + injected faults (target=1)
    test_frames = [clean_test]
    if not fault_test.empty:
        test_frames.append(fault_test)
    test_combined = pd.concat(test_frames, ignore_index=True)

    x_test = safe_numeric_frame(test_combined, builder.feature_list)
    y_test = test_combined["target"].fillna(0).astype(int)

    model_results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}

    for model_name, parameter_list in build_candidate_models(
        random_state=config.random_state,
    ).items():
        best_model = None
        best_val_metrics = None
        best_params = None
        best_threshold = 0.5

        for params in parameter_list:
            try:
                started = time.perf_counter()
                candidate = create_model(
                    model_name,
                    params,
                    random_state=config.random_state,
                )
                candidate.fit(x_train)
                train_duration = time.perf_counter() - started

                # Calibration search for operating threshold on validation set
                val_scores = candidate.score_samples(x_val)
                potential_thresholds = np.linspace(0.3, 0.8, 11)
                best_sub_metrics = None
                best_sub_thresh = 0.5
                best_sub_score = float("-inf")

                for th in potential_thresholds:
                    m = evaluate_detector(
                        candidate,
                        x_val,
                        y_val,
                        metadata=val_combined,
                        threshold=th,
                    )
                    # Objective: prioritize high F1 and high Specificity (low FPR)
                    score = m["f1"] * 0.4 + m["balanced_accuracy"] * 0.4 - m["false_positive_rate"] * 0.2
                    if score > best_sub_score:
                        best_sub_score = score
                        best_sub_metrics = m
                        best_sub_thresh = float(th)

                if best_sub_metrics is not None:
                    best_sub_metrics["training_time"] = train_duration
                    best_sub_metrics["calibrated_threshold"] = best_sub_thresh

                    if (
                        best_val_metrics is None
                        or best_sub_score > (
                            best_val_metrics["f1"] * 0.4
                            + best_val_metrics["balanced_accuracy"] * 0.4
                            - best_val_metrics["false_positive_rate"] * 0.2
                        )
                    ):
                        best_model = candidate
                        best_val_metrics = best_sub_metrics
                        best_params = params
                        best_threshold = best_sub_thresh

            except Exception as exc:
                model_results.setdefault(model_name, {}).setdefault("errors", []).append(str(exc))

        if best_model is None or best_val_metrics is None:
            continue

        trained_models[model_name] = best_model

        # Evaluate on untouched test set using calibrated threshold
        test_metrics = evaluate_detector(
            best_model,
            x_test,
            y_test,
            metadata=test_combined,
            threshold=best_threshold,
        )

        model_results[model_name] = {
            "hyperparameters": best_params,
            "calibrated_threshold": best_threshold,
            "validation_metrics": best_val_metrics,
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
        "calibrated_threshold": selected_result.get("calibrated_threshold", 0.5),
        "feature_builder": builder,
        "baseline": baseline,
        "config": config,
        "version": "2.0.0",
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
        "model_version": "2.0.0",
        "hyperparameters": selected_result["hyperparameters"],
        "calibrated_threshold": selected_result.get("calibrated_threshold", 0.5),
        "training_period": {
            "start": str(usable["timestamp"].min()),
            "end": str(usable["timestamp"].max()),
        },
        "validation_metrics": selected_result["validation_metrics"],
        "test_metrics": selected_result["test_metrics"],
        "feature_list": builder.feature_list,
        "feature_count": len(builder.feature_list),
        "selection_score": selected_result["selection_score"],
        "validation_report": {
            "row_count": validation_report.row_count,
            "valid_row_count": validation_report.valid_row_count,
            "invalid_row_count": validation_report.invalid_row_count,
            "summary": validation_report.summary,
        },
        "autoencoder": "NOT_USED",
        "confidence_status": "CALIBRATED_V2",
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
