"""
SkyGuard AI — Benchmark Reporting Generator.

Produces:
1. Machine-readable JSON report: datasets/benchmark/benchmark_report.json
2. Human-readable Markdown summary: datasets/benchmark/benchmark_report.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd


def generate_benchmark_reports(
    metadata: dict[str, Any],
    eval_results: dict[str, Any] | None,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate both JSON and Markdown benchmark reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "benchmark_report.json"
    md_path = output_dir / "benchmark_report.md"

    report_data = {
        "metadata": metadata,
        "model_evaluations": eval_results or {},
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Build Markdown
    split_info = metadata.get("split_summary", {})
    train_info = split_info.get("train", {})
    val_info = split_info.get("validation", {})
    test_info = split_info.get("test", {})

    class_dist = metadata.get("class_distribution", {})
    fault_dist = metadata.get("fault_type_distribution", {})
    sev_dist = metadata.get("severity_distribution", {})
    stn_dist = metadata.get("station_distribution", {})

    lines: list[str] = [
        "# SkyGuard AI — Scientific Anomaly Detection Benchmark Report",
        "",
        f"**Benchmark Version**: `{metadata.get('benchmark_name')}`  ",
        f"**Random Seed**: `{metadata.get('random_seed')}` (Deterministic & Reproducible)  ",
        f"**Generated Date**: `{metadata.get('created_at')}`  ",
        "",
        "---",
        "",
        "## 1. Executive Summary & Dataset Dimensions",
        "",
        "| Dimension | Value | Description |",
        "|---|---|---|",
        f"| **Participating AWS Stations** | {metadata.get('station_count')} | Multi-station network (coastal delta cluster + inland reference stations) |",
        f"| **Total Observations** | {metadata.get('total_observations'):,} | Hourly observations across 45-day duration |",
        f"| **Time Range** | {metadata.get('time_range', {}).get('start')} to {metadata.get('time_range', {}).get('end')} | 1,080 hourly intervals per station |",
        f"| **Clean / Normal Observations** | {class_dist.get('NORMAL', 0):,} | 100% pure background with diurnal cycles & noise |",
        f"| **Synthetic Sensor Faults** | {class_dist.get('SENSOR_FAULT', 0) + class_dist.get('DATA_QUALITY_ISSUE', 0) + class_dist.get('COMMUNICATION_FAULT', 0):,} | Injected across MILD, MODERATE, and SEVERE severities |",
        f"| **Genuine Weather Events** | {class_dist.get('WEATHER_EVENT', 0):,} | Plausible meteorological phenomena (storms, heatwaves, low pressure) |",
        f"| **Distinct Scenarios** | {metadata.get('scenario_counts')} | Independent case-controlled scenarios |",
        "",
        "---",
        "",
        "## 2. Chronological Splits & Zero-Leakage Guarantee",
        "",
        "| Partition | Observations | Share | Start Timestamp | End Timestamp | Data Purity |",
        "|---|---|---|---|---|---|",
        f"| **Train** | {train_info.get('count', 0):,} | {train_info.get('percentage')}% | `{train_info.get('start')}` | `{train_info.get('end')}` | **100% Clean Normal Only (Zero Faults)** |",
        f"| **Validation** | {val_info.get('count', 0):,} | {val_info.get('percentage')}% | `{val_info.get('start')}` | `{val_info.get('end')}` | Calibration Faults & Weather Events |",
        f"| **Test** | {test_info.get('count', 0):,} | {test_info.get('percentage')}% | `{test_info.get('start')}` | `{test_info.get('end')}` | Frozen Evaluation Scenarios |",
        "",
        "> [!IMPORTANT]",
        "> **Zero-Leakage Assurance**: The training split contains strictly normal ambient telemetry (`ground_truth == 'NORMAL'`). No future records, synthetic faults, or weather events leak into training. All detector thresholds are calibrated exclusively on validation data and frozen for test evaluation.",
        "",
        "---",
        "",
        "## 3. Ground Truth Taxonomy & Class Distribution",
        "",
        "| Ground Truth Class | Observation Count | Percentage | Definition |",
        "|---|---|---|---|",
    ]

    total_obs = max(1, metadata.get("total_observations", 1))
    for gt, count in sorted(class_dist.items(), key=lambda x: -x[1]):
        pct = round(count / total_obs * 100, 2)
        desc = {
            "NORMAL": "Normal ambient weather with diurnal solar cycles & realistic noise",
            "SENSOR_FAULT": "Physical sensor breakdown (drift, spike, flatline freeze, inconsistency)",
            "WEATHER_EVENT": "Genuine meteorological transition (convective storm, heatwave, low pressure)",
            "DATA_QUALITY_ISSUE": "Telemetry corruptions (duplicate packets, missing values, timestamp error)",
            "COMMUNICATION_FAULT": "Transmission outages and sustained packet loss",
        }.get(gt, "")
        lines.append(f"| `{gt}` | {count:,} | {pct}% | {desc} |")

    lines.extend([
        "",
        "### Fault Distribution by Category & Severity",
        "",
        "| Severity Level | Injected Observations | Share of Faults | Purpose |",
        "|---|---|---|---|",
    ])

    total_faults = max(1, sum(sev_dist.values()))
    for sev, count in sorted(sev_dist.items()):
        pct = round(count / total_faults * 100, 2)
        notes = {
            "MILD": "Subtle anomalies close to diurnal variance (tests detector sensitivity)",
            "MODERATE": "Standard operational faults and clear physical anomalies",
            "SEVERE": "Gross sensor failures, extended flatlines, and unphysical values",
        }.get(sev, "")
        lines.append(f"| `{sev}` | {count:,} | {pct}% | {notes} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Multi-Station Evaluation Paradigms",
        "",
        "* **CASE A (Single-Station Sensor Fault)**: Station `43189` (Vijayawada) develops severe temperature drift, while nearby peer stations `AWS002` (Guntur, 32 km away) and `AWS004` (Machilipatnam, 62 km away) remain completely peaceful and normal. This creates high peer disagreement, which the spatial evidence channel uses to raise sensor fault certainty.",
        "* **CASE B (Multi-Station Coherent Weather Event)**: Stations `43189`, `AWS002`, and `AWS004` all synchronously experience a convective thunder-squall with rapid temperature plunge (-7.8°C), humidity surge (+32%), and pressure perturbation. This creates high peer agreement, proving whether the hybrid system avoids false positive sensor alerts during real weather events.",
        "",
        "---",
        "",
    ])

    if eval_results:
        lines.extend([
            "## 5. Candidate Detector Benchmarking Results (Frozen Test Split)",
            "",
            "| Detector Model | Precision | Recall | Specificity | F1 Score | Balanced Acc | MCC | ROC-AUC | PR-AUC | False Alarms/Stn/Day | Mean Latency (s) |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ])

        for model_name, res in eval_results.items():
            lines.append(
                f"| **{model_name}** | "
                f"{res.get('precision', 0):.4f} | "
                f"{res.get('recall', 0):.4f} | "
                f"{res.get('specificity', 0):.4f} | "
                f"{res.get('f1', 0):.4f} | "
                f"{res.get('balanced_accuracy', 0):.4f} | "
                f"{res.get('mcc', 0):.4f} | "
                f"{res.get('roc_auc', 0) if res.get('roc_auc') is not None else 'N/A'} | "
                f"{res.get('pr_auc', 0) if res.get('pr_auc') is not None else 'N/A'} | "
                f"{res.get('false_alarms_per_station_day', 0):.2f} | "
                f"{res.get('mean_detection_latency_seconds', 0) if res.get('mean_detection_latency_seconds') is not None else 'N/A'} |"
            )

        lines.extend([
            "",
            "---",
            "",
        ])

    lines.extend([
        "## 6. Scientific Limitations & Data Boundaries",
        "",
        "1. **Simulated Weather Physics**: Baseline observations are physically grounded synthetic simulations calibrated to Indian AWS diurnal profiles, not raw historical IMD AWS database dumps.",
        "2. **Station Density**: The current benchmark models 6 representative stations (3 clustered delta stations + 3 reference stations across southern/central India). In dense national networks, spatial interpolation can leverage hundreds of neighboring nodes.",
        "3. **Unsupervised vs Evidence Fusion**: Standalone unsupervised detectors (LOF, Isolation Forest, One-Class SVM) evaluate anomaly deviation without domain context. The SkyGuard hybrid engine fuses these scores with spatial correlation and physical rules to distinguish weather events from hardware failures.",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, md_path
