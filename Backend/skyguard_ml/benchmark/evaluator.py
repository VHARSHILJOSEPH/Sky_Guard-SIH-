"""
SkyGuard AI — Scientific Benchmark Evaluator.

Executes standardized evaluation of candidate detectors:
1. LOF (Local Outlier Factor)
2. Isolation Forest
3. One-Class SVM
4. SkyGuard Rule / Statistical Detectors
5. SkyGuard Hybrid Evidence-Fusion Pipeline

Enforces the strict scientific evaluation protocol:
- Fit models on CLEAN TRAINING DATA ONLY.
- Calibrate thresholds on VALIDATION DATA ONLY.
- Freeze threshold for TEST DATA EVALUATION.
- Never use test data for tuning, calibration, or model decisions.
- Compute all 12 required metrics:
  Precision, Recall, F1, Specificity, FPR, FNR, Balanced Accuracy,
  MCC, ROC-AUC, PR-AUC, False Alarms/Station/Day, and True Detection Latency.
- Compute fault-wise and weather-event discrimination metrics.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from skyguard_lof import build_feature_dataset, SkyGuardLOFConfig, FEATURE_NAMES


def compute_comprehensive_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    raw_scores: Sequence[float] | None = None,
    timestamps: Sequence[str] | None = None,
    n_stations: int = 1,
) -> dict[str, Any]:
    """Compute all 12 standard metrics conforming to SkyGuard specification."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    ).ravel()

    precision = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    balanced_acc = float((recall + specificity) / 2.0)

    try:
        mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
    except Exception:
        mcc = 0.0

    roc_auc: float | None = None
    pr_auc: float | None = None

    if raw_scores is not None and len(np.unique(y_true_arr)) >= 2:
        try:
            roc_auc = float(roc_auc_score(y_true_arr, raw_scores))
            pr_auc = float(average_precision_score(y_true_arr, raw_scores))
        except Exception:
            pass

    # True false alarms per station day
    false_alarms_per_station_day: float | None = None
    if timestamps is not None and len(timestamps) > 1:
        ts_objs = pd.to_datetime(timestamps, utc=True)
        duration_days = max(1.0 / 24.0, (ts_objs.max() - ts_objs.min()).total_seconds() / 86400.0)
        false_alarms_per_station_day = float(fp / (duration_days * max(1, n_stations)))

    return {
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "false_alarms_per_station_day": false_alarms_per_station_day,
    }


def compute_detection_latency(
    test_df: pd.DataFrame,
    predictions: Sequence[int],
) -> dict[str, float | None]:
    """
    Compute actual detection latency in seconds from event_start to first positive detection
    for each distinct scenario.
    """
    latencies: dict[str, float | None] = {}
    preds = np.asarray(predictions)

    for scenario_id, group in test_df.groupby("scenario_id"):
        if scenario_id == "SCN_CLEAN_BASELINE":
            continue

        start_time_val = group["event_start"].dropna().iloc[0] if not group["event_start"].dropna().empty else None
        if not start_time_val:
            continue

        event_start_dt = pd.Timestamp(start_time_val)
        group_indices = group.index.to_numpy()
        group_preds = preds[group_indices]

        # Find first prediction == 1 at or after event_start
        detected_mask = group_preds == 1
        if detected_mask.any():
            detected_rows = group.iloc[np.where(detected_mask)[0]]
            detected_dts = pd.to_datetime(detected_rows["timestamp"], utc=True)
            valid_detects = detected_dts[detected_dts >= event_start_dt]
            if not valid_detects.empty:
                latency_sec = float((valid_detects.min() - event_start_dt).total_seconds())
                latencies[scenario_id] = latency_sec
            else:
                latencies[scenario_id] = None
        else:
            latencies[scenario_id] = None

    return latencies


class BenchmarkEvaluator:
    """
    Benchmarks unsupervised anomaly detectors against the standardized benchmark.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.config = SkyGuardLOFConfig(min_history_length=24, rolling_window_hours=24)

    def _prepare_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Convert telemetry dataframe to causal rolling features."""
        records = df.to_dict(orient="records")
        dataset = build_feature_dataset(records, config=self.config)
        X = dataset.X.to_numpy(dtype=float)
        y = np.asarray([int(r.get("is_anomaly", 0)) for r in dataset.records], dtype=int)
        return X, y, dataset.records

    def evaluate_detector(
        self,
        model_name: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Full scientific workflow:
        1. Train model on clean train_df ONLY.
        2. Calibrate threshold on val_df ONLY.
        3. Evaluate frozen model on test_df ONLY.
        """
        started = time.perf_counter()

        # Verify clean training data purity
        if (train_df["is_anomaly"] != 0).any():
            raise ValueError("Training data contains anomaly labels! Training must be strictly clean normal only.")

        X_train, _, _ = self._prepare_features(train_df)
        X_val, y_val, val_records = self._prepare_features(val_df)
        X_test, y_test, test_records = self._prepare_features(test_df)

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_val_imp = imputer.transform(X_val)
        X_test_imp = imputer.transform(X_test)

        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train_imp)
        X_val_sc = scaler.transform(X_val_imp)
        X_test_sc = scaler.transform(X_test_imp)

        # Instantiate model
        if model_name == "LOF":
            estimator = LocalOutlierFactor(
                n_neighbors=20,
                contamination="auto",
                novelty=True,
                n_jobs=-1,
            )
            estimator.fit(X_train_sc)
            raw_val_scores = -estimator.decision_function(X_val_sc)
            raw_test_scores = -estimator.decision_function(X_test_sc)

        elif model_name == "IsolationForest":
            estimator = IsolationForest(
                n_estimators=150,
                contamination="auto",
                random_state=self.random_state,
                n_jobs=-1,
            )
            estimator.fit(X_train_sc)
            raw_val_scores = -estimator.score_samples(X_val_sc)
            raw_test_scores = -estimator.score_samples(X_test_sc)

        elif model_name == "OneClassSVM":
            estimator = OneClassSVM(
                kernel="rbf",
                nu=0.05,
                gamma="scale",
            )
            estimator.fit(X_train_sc)
            raw_val_scores = -estimator.decision_function(X_val_sc)
            raw_test_scores = -estimator.decision_function(X_test_sc)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        training_time = time.perf_counter() - started

        # -------------------------------------------------------------
        # Threshold Calibration (VALIDATION DATA ONLY)
        # -------------------------------------------------------------
        candidates = np.unique(raw_val_scores)
        best_thresh = float(np.median(candidates))
        best_val_f1 = -1.0

        for cand in candidates:
            preds = (raw_val_scores >= cand).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_val_f1:
                best_val_f1 = f1
                best_thresh = float(cand)

        # -------------------------------------------------------------
        # Frozen Threshold Test Evaluation (TEST DATA ONLY)
        # -------------------------------------------------------------
        test_predictions = (raw_test_scores >= best_thresh).astype(int)

        test_timestamps = [r["timestamp"] for r in test_records]
        n_stations = len(set(r["station_id"] for r in test_records))

        overall_metrics = compute_comprehensive_metrics(
            y_true=y_test,
            y_pred=test_predictions,
            raw_scores=raw_test_scores,
            timestamps=test_timestamps,
            n_stations=n_stations,
        )
        overall_metrics["calibrated_threshold"] = best_thresh
        overall_metrics["validation_f1"] = best_val_f1
        overall_metrics["training_time_seconds"] = training_time

        # Compute per-scenario detection latencies
        test_records_df = pd.DataFrame(test_records)
        latencies = compute_detection_latency(test_records_df, test_predictions)
        overall_metrics["detection_latencies_seconds"] = latencies
        valid_lats = [v for v in latencies.values() if v is not None]
        overall_metrics["mean_detection_latency_seconds"] = (
            float(np.mean(valid_lats)) if valid_lats else None
        )

        # Fault-wise breakdown
        fault_wise: dict[str, Any] = {}
        for fault_type in sorted(set(r.get("fault_type", "NONE") for r in test_records)):
            indices = [i for i, r in enumerate(test_records) if r.get("fault_type") == fault_type]
            if indices:
                sub_true = y_test[indices]
                sub_pred = test_predictions[indices]
                sub_scores = raw_test_scores[indices]
                fault_wise[fault_type] = {
                    "count": len(indices),
                    "precision": float(precision_score(sub_true, sub_pred, zero_division=0)),
                    "recall": float(recall_score(sub_true, sub_pred, zero_division=0)),
                    "f1": float(f1_score(sub_true, sub_pred, zero_division=0)),
                }
        overall_metrics["fault_wise_metrics"] = fault_wise

        # Weather event false alarm analysis
        wx_indices = [i for i, r in enumerate(test_records) if r.get("ground_truth") == "WEATHER_EVENT"]
        if wx_indices:
            wx_preds = test_predictions[wx_indices]
            # How often did the model call a weather event an anomaly?
            overall_metrics["weather_event_alarm_rate"] = float(np.mean(wx_preds == 1))
        else:
            overall_metrics["weather_event_alarm_rate"] = None

        return overall_metrics

    def run_benchmark_suite(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run standardized evaluation across LOF, Isolation Forest, and One-Class SVM."""
        results: dict[str, Any] = {}
        for name in ("LOF", "IsolationForest", "OneClassSVM"):
            results[name] = self.evaluate_detector(
                model_name=name,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
            )
        return results
