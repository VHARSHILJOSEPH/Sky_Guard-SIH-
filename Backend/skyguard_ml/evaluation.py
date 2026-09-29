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
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)


def evaluate_detector(
    model: Any,
    x: pd.DataFrame,
    y: pd.Series | np.ndarray,
    metadata: pd.DataFrame | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()

    scores = model.score_samples(x)
    if threshold is not None:
        predictions = np.where(scores >= threshold, 1, 0)
    else:
        predictions = model.predict(x)

    inference_time = time.perf_counter() - start

    y_true = np.asarray(y).astype(int)
    y_pred = np.asarray(predictions).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )
    recall = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )
    f1 = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    total_negatives = max(tn + fp, 1)
    specificity = float(tn / total_negatives)
    false_positive_rate = float(fp / total_negatives)
    balanced_accuracy = float(0.5 * (recall + specificity))

    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = 0.0

    # ROC-AUC & PR-AUC require at least one sample of each class
    roc_auc: float | None = None
    pr_auc: float | None = None
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, scores))
            pr_auc = float(average_precision_score(y_true, scores))
        except Exception:
            pass

    # False alarms per station day (assuming 24 hourly observations per day for AWS)
    false_alarms_per_station_day = float(false_positive_rate * 24.0)

    # Detection latency: time/steps from fault onset to first detection
    latencies: list[int] = []
    in_fault = False
    fault_start_idx = 0
    detected_in_fault = False

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if not in_fault:
                in_fault = True
                fault_start_idx = i
                detected_in_fault = False
            if y_pred[i] == 1 and not detected_in_fault:
                latencies.append(i - fault_start_idx)
                detected_in_fault = True
        else:
            in_fault = False

    detection_latency = float(np.mean(latencies)) if latencies else None

    # Fault-wise breakdown if metadata is available
    fault_breakdown: dict[str, dict[str, Any]] = {}
    if metadata is not None and "fault_type" in metadata.columns:
        for fault_type in metadata["fault_type"].dropna().unique():
            if fault_type == "NORMAL":
                continue
            mask = metadata["fault_type"] == fault_type
            f_true = y_true[mask]
            f_pred = y_pred[mask]
            if len(f_true) > 0:
                f_recall = float(recall_score(f_true, f_pred, zero_division=0))
                fault_breakdown[str(fault_type)] = {
                    "count": int(len(f_true)),
                    "detected": int(np.sum((f_true == 1) & (f_pred == 1))),
                    "recall": f_recall,
                }

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_positive_rate": false_positive_rate,
        "false_alarms_per_station_day": false_alarms_per_station_day,
        "detection_latency": detection_latency,
        "training_time": None,
        "inference_time": inference_time,
        "mean_anomaly_score": float(np.mean(scores)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "sample_count": len(y_true),
        "fault_breakdown": fault_breakdown,
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
        y_t = np.asarray(y_true).astype(int)
        y_p = np.asarray(predictions).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)

        rows.append(
            {
                "system": name,
                "precision": float(precision_score(y_t, y_p, zero_division=0)),
                "recall": float(recall_score(y_t, y_p, zero_division=0)),
                "f1": float(f1_score(y_t, y_p, zero_division=0)),
                "specificity": float(tn / max(tn + fp, 1)),
                "false_positive_rate": float(fpr),
                "false_alarms_per_day": float(fpr * 24.0),
            }
        )

    return pd.DataFrame(rows)
