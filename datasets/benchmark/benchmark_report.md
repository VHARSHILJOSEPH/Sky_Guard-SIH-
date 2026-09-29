# SkyGuard AI — Scientific Anomaly Detection Benchmark Report

**Benchmark Version**: `SkyGuard-Scientific-Benchmark-v1.0`  
**Random Seed**: `42` (Deterministic & Reproducible)  
**Generated Date**: `2026-09-14T15:11:48.419491+00:00`  

---

## 1. Executive Summary & Dataset Dimensions

| Dimension | Value | Description |
|---|---|---|
| **Participating AWS Stations** | 6 | Multi-station network (coastal delta cluster + inland reference stations) |
| **Total Observations** | 6,480 | Hourly observations across 45-day duration |
| **Time Range** | 2026-08-01T00:00:00Z to 2026-09-14T23:00:00Z | 1,080 hourly intervals per station |
| **Clean / Normal Observations** | 6,164 | 100% pure background with diurnal cycles & noise |
| **Synthetic Sensor Faults** | 118 | Injected across MILD, MODERATE, and SEVERE severities |
| **Genuine Weather Events** | 198 | Plausible meteorological phenomena (storms, heatwaves, low pressure) |
| **Distinct Scenarios** | 23 | Independent case-controlled scenarios |

---

## 2. Chronological Splits & Zero-Leakage Guarantee

| Partition | Observations | Share | Start Timestamp | End Timestamp | Data Purity |
|---|---|---|---|---|---|
| **Train** | 3,888 | 60.0% | `2026-08-01T00:00:00Z` | `2026-08-27T23:00:00Z` | **100% Clean Normal Only (Zero Faults)** |
| **Validation** | 1,296 | 20.0% | `2026-08-28T00:00:00Z` | `2026-09-05T23:00:00Z` | Calibration Faults & Weather Events |
| **Test** | 1,296 | 20.0% | `2026-09-06T00:00:00Z` | `2026-09-14T23:00:00Z` | Frozen Evaluation Scenarios |

> [!IMPORTANT]
> **Zero-Leakage Assurance**: The training split contains strictly normal ambient telemetry (`ground_truth == 'NORMAL'`). No future records, synthetic faults, or weather events leak into training. All detector thresholds are calibrated exclusively on validation data and frozen for test evaluation.

---

## 3. Ground Truth Taxonomy & Class Distribution

| Ground Truth Class | Observation Count | Percentage | Definition |
|---|---|---|---|
| `NORMAL` | 6,164 | 95.12% | Normal ambient weather with diurnal solar cycles & realistic noise |
| `WEATHER_EVENT` | 198 | 3.06% | Genuine meteorological transition (convective storm, heatwave, low pressure) |
| `SENSOR_FAULT` | 89 | 1.37% | Physical sensor breakdown (drift, spike, flatline freeze, inconsistency) |
| `DATA_QUALITY_ISSUE` | 21 | 0.32% | Telemetry corruptions (duplicate packets, missing values, timestamp error) |
| `COMMUNICATION_FAULT` | 8 | 0.12% | Transmission outages and sustained packet loss |

### Fault Distribution by Category & Severity

| Severity Level | Injected Observations | Share of Faults | Purpose |
|---|---|---|---|
| `MILD` | 35 | 11.08% | Subtle anomalies close to diurnal variance (tests detector sensitivity) |
| `MODERATE` | 125 | 39.56% | Standard operational faults and clear physical anomalies |
| `SEVERE` | 156 | 49.37% | Gross sensor failures, extended flatlines, and unphysical values |

---

## 4. Multi-Station Evaluation Paradigms

* **CASE A (Single-Station Sensor Fault)**: Station `43189` (Vijayawada) develops severe temperature drift, while nearby peer stations `AWS002` (Guntur, 32 km away) and `AWS004` (Machilipatnam, 62 km away) remain completely peaceful and normal. This creates high peer disagreement, which the spatial evidence channel uses to raise sensor fault certainty.
* **CASE B (Multi-Station Coherent Weather Event)**: Stations `43189`, `AWS002`, and `AWS004` all synchronously experience a convective thunder-squall with rapid temperature plunge (-7.8°C), humidity surge (+32%), and pressure perturbation. This creates high peer agreement, proving whether the hybrid system avoids false positive sensor alerts during real weather events.

---

## 5. Candidate Detector Benchmarking Results (Frozen Test Split)

| Detector Model | Precision | Recall | Specificity | F1 Score | Balanced Acc | MCC | ROC-AUC | PR-AUC | False Alarms/Stn/Day | Mean Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **LOF** | 0.5299 | 0.8590 | 0.8130 | 0.6555 | 0.8360 | 0.5733 | 0.8716751994285035 | 0.5667143319398688 | 3.62 | 6872.727272727273 |
| **IsolationForest** | 0.4938 | 0.7004 | 0.8238 | 0.5792 | 0.7621 | 0.4647 | 0.7905655435170854 | 0.510489515455755 | 3.41 | 4114.285714285715 |
| **OneClassSVM** | 0.6377 | 0.7445 | 0.8962 | 0.6870 | 0.8204 | 0.6056 | 0.8259340397666387 | 0.5670571229487616 | 2.01 | 4114.285714285715 |

---

## 6. Scientific Limitations & Data Boundaries

1. **Simulated Weather Physics**: Baseline observations are physically grounded synthetic simulations calibrated to Indian AWS diurnal profiles, not raw historical IMD AWS database dumps.
2. **Station Density**: The current benchmark models 6 representative stations (3 clustered delta stations + 3 reference stations across southern/central India). In dense national networks, spatial interpolation can leverage hundreds of neighboring nodes.
3. **Unsupervised vs Evidence Fusion**: Standalone unsupervised detectors (LOF, Isolation Forest, One-Class SVM) evaluate anomaly deviation without domain context. The SkyGuard hybrid engine fuses these scores with spatial correlation and physical rules to distinguish weather events from hardware failures.