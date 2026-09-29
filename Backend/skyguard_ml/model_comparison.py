from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baselines import BaselineStore
from .config import TrainingConfig
from .evaluation import evaluate_detector
from .feature_engineering import BASE_VARIABLES, FeatureBuilder
from .model_selection import calculate_selection_score, select_best_model
from .models import build_candidate_models, create_model
from .preprocessing import chronological_split, safe_numeric_frame
from .schemas import observations_to_frame
from .synthetic_faults import generate_synthetic_faults
from .validation import validate_observations


def run_model_comparison(
    observations: list[dict[str, Any]] | pd.DataFrame,
    config: TrainingConfig | None = None,
    output_dir: str | Path = "skyguard_ml/artifacts",
) -> dict[str, Any]:
    """
    Run fair, comprehensive comparison across all 4 candidate models:
    - LOF (Local Outlier Factor)
    - Isolation Forest
    - One-Class SVM
    - Robust Covariance (Elliptic Envelope)

    Ensures identical training data, feature engineering, and validation/test splits.
    """
    config = config or TrainingConfig()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_frame = observations_to_frame(observations)
    validated_frame, validation_report = validate_observations(raw_frame)

    usable = validated_frame[
        validated_frame["station_id"].notna()
        & validated_frame["timestamp"].notna()
    ].copy()

    if len(usable) < config.min_training_rows:
        raise ValueError(
            f"At least {config.min_training_rows} usable rows required, got {len(usable)}."
        )

    # 1. Fit baselines and build features
    baseline = BaselineStore(minimum_samples=config.feature.minimum_baseline_samples)
    baseline.fit(usable, BASE_VARIABLES)

    builder = FeatureBuilder(config.feature)
    clean_features = builder.fit_transform(usable, baseline=baseline)
    clean_features["target"] = 0
    clean_features["fault_type"] = "NORMAL"

    clean_train, clean_val, clean_test = chronological_split(
        clean_features,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    # 2. Generate synthetic faults for validation & testing
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

    _, fault_val, fault_test = chronological_split(
        fault_features,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    # Train: clean-only
    x_train = safe_numeric_frame(clean_train, builder.feature_list)

    # Validation: clean + fault
    val_combined = pd.concat([clean_val, fault_val], ignore_index=True)
    x_val = safe_numeric_frame(val_combined, builder.feature_list)
    y_val = val_combined["target"].fillna(0).astype(int)

    # Test: clean + fault (untouched until final evaluation)
    test_combined = pd.concat([clean_test, fault_test], ignore_index=True)
    x_test = safe_numeric_frame(test_combined, builder.feature_list)
    y_test = test_combined["target"].fillna(0).astype(int)

    candidates = build_candidate_models(random_state=config.random_state)
    comparison_results: dict[str, dict[str, Any]] = {}
    table_rows: list[dict[str, Any]] = []

    for model_name, param_list in candidates.items():
        best_model = None
        best_val_metrics = None
        best_params = None
        best_threshold = 0.5
        best_composite = float("-inf")
        errors: list[str] = []

        for params in param_list:
            try:
                t0 = time.perf_counter()
                model = create_model(model_name, params, random_state=config.random_state)
                model.fit(x_train)
                fit_time = time.perf_counter() - t0

                # Calibration search for operating threshold
                thresholds = np.linspace(0.3, 0.8, 11)
                for th in thresholds:
                    val_metrics = evaluate_detector(
                        model,
                        x_val,
                        y_val,
                        metadata=val_combined,
                        threshold=th,
                    )
                    score = calculate_selection_score(val_metrics, config.selection_weights.as_dict())
                    if score > best_composite:
                        best_composite = score
                        best_model = model
                        best_val_metrics = val_metrics
                        best_val_metrics["training_time"] = fit_time
                        best_val_metrics["calibrated_threshold"] = float(th)
                        best_params = params
                        best_threshold = float(th)
            except Exception as e:
                errors.append(str(e))

        if best_model is None or best_val_metrics is None:
            comparison_results[model_name] = {
                "status": "FAILED",
                "errors": errors,
            }
            continue

        # Evaluate on untouched test set
        test_metrics = evaluate_detector(
            best_model,
            x_test,
            y_test,
            metadata=test_combined,
            threshold=best_threshold,
        )

        comparison_results[model_name] = {
            "status": "SUCCESS",
            "hyperparameters": best_params,
            "calibrated_threshold": best_threshold,
            "selection_score": best_composite,
            "validation_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        }

        table_rows.append({
            "Model": model_name,
            "F1": round(test_metrics["f1"], 4),
            "Balanced Acc": round(test_metrics["balanced_accuracy"], 4),
            "Specificity": round(test_metrics["specificity"], 4),
            "Precision": round(test_metrics["precision"], 4),
            "Recall": round(test_metrics["recall"], 4),
            "MCC": round(test_metrics["mcc"], 4),
            "ROC-AUC": round(test_metrics["roc_auc"], 4) if test_metrics["roc_auc"] is not None else "N/A",
            "FPR": round(test_metrics["false_positive_rate"], 4),
            "False Alarms/Day": round(test_metrics["false_alarms_per_station_day"], 2),
            "Latency (steps)": round(test_metrics["detection_latency"], 2) if test_metrics["detection_latency"] is not None else "N/A",
            "Train Time (s)": round(best_val_metrics["training_time"], 3),
            "Infer Time (ms)": round(test_metrics["inference_time"] * 1000, 2),
            "Selection Score": round(best_composite, 4),
        })

    # Determine winner based on validation selection score
    successful = {k: v for k, v in comparison_results.items() if v.get("status") == "SUCCESS"}
    if not successful:
        raise RuntimeError("No candidate model completed successfully.")

    best_name, best_info = select_best_model(successful, config.selection_weights.as_dict())

    def _make_md_table(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
        return "\n".join(lines)

    df_table = pd.DataFrame(table_rows).sort_values("Selection Score", ascending=False)
    markdown_table = _make_md_table(df_table)

    report_payload = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "winning_model": best_name,
        "feature_count": len(builder.feature_list),
        "clean_train_rows": len(x_train),
        "validation_rows": len(x_val),
        "test_rows": len(x_test),
        "comparison_table": table_rows,
        "model_results": comparison_results,
    }

    # Save outputs
    (out_path / "model_comparison_report.json").write_text(
        json.dumps(report_payload, indent=2, default=str)
    )

    md_report = (
        f"# SkyGuard AI — Phase 8 ML Model Comparison Report\n\n"
        f"**Selected Model**: `{best_name}`\n"
        f"**Selection Criterion**: Composite score prioritizing low FPR, high balanced accuracy, and robust specificity.\n"
        f"**Features Evaluated**: {len(builder.feature_list)} features (SIH core T/P/RH only).\n\n"
        f"## Final Candidate Comparison Table (Evaluated on Untouched Test Set)\n\n"
        f"{markdown_table}\n\n"
        f"## Selection Justification\n\n"
        f"`{best_name}` achieved the highest composite validation score ({round(best_info['selection_score'], 4)}) "
        f"with test balanced accuracy of {round(best_info['test_metrics']['balanced_accuracy'], 4)} "
        f"and false positive rate of {round(best_info['test_metrics']['false_positive_rate'], 4)}.\n"
    )
    (out_path / "model_comparison_report.md").write_text(md_report)

    return {
        "winning_model": best_name,
        "comparison_table": table_rows,
        "markdown_table": markdown_table,
        "report": report_payload,
    }


if __name__ == "__main__":
    from .training import TrainingConfig
    # Run comparison on synthetic clean dataset by default
    print("Running SkyGuard ML Model Comparison...")
