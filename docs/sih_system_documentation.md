# SkyGuard AI — Comprehensive SIH System Documentation

## Operational Weather Sensor Quality Intelligence & Automated Anomaly Detection Platform

---

### Table of Contents

1. [Executive Summary & Problem Formulation](#1-executive-summary--problem-formulation)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [7-Channel Evidence Fusion Framework](#3-7-channel-evidence-fusion-framework)
4. [Authoritative Machine Learning Model & Comparative Benchmark](#4-authoritative-machine-learning-model--comparative-benchmark)
5. [Multi-Signal Station Health Governance](#5-multi-signal-station-health-governance)
6. [Operational Anomaly Lifecycle Management](#6-operational-anomaly-lifecycle-management)
7. [Hardware-in-the-Loop (HIL) Testbed Architecture](#7-hardware-in-the-loop-hil-testbed-architecture)
8. [Data Provenance & Scientific Integrity Standards](#8-data-provenance--scientific-integrity-standards)
9. [Smart India Hackathon (SIH) Demonstration Runbook](#9-smart-india-hackathon-sih-demonstration-runbook)

---

### 1. Executive Summary & Problem Formulation

Automatic Weather Stations (AWS) operated by meteorological agencies like the India Meteorological Department (IMD) continuously transmit observations critical to disaster management, aviation, agriculture, and numerical weather prediction. However, automated networks encounter unavoidable data quality disruptions:

- **Transducer calibration drift**: Slow cumulative sensor degradation departing from diurnal physical cycles.
- **Physical sensor freezing**: Solid-state ADC lockup yielding identical readings with zero variance.
- **Atmospheric rate-of-change violations**: Electrical spikes that exceed physical environmental rate limits.
- **Communication packet dropouts**: Transmission dropouts leading to corrupted or missing fields.
- **Multivariate thermodynamic decoupling**: Inconsistent atmospheric combinations (e.g., 100% relative humidity alongside large dewpoint depressions).

#### The Core Scientific Dilemma

A conventional fixed-threshold system cannot reliably determine whether a sudden shift is an **isolated sensor transducer failure** or a **genuine extreme meteorological event** (such as a thunderstorm gust front or microburst). Discarding valid extreme events endangers public safety, while passing faulty sensor data pollutes numerical forecasting models.

**SkyGuard AI** addresses this challenge through an explainable, multi-source evidence fusion intelligence platform. It fuses deterministic physical data-quality validation, rolling diurnal statistical baselines, compact unsupervised density modeling, multivariate thermodynamic physics, regional spatial peer consensus, and synoptic weather context into an authoritative, auditable decision stream.

---

### 2. End-to-End System Architecture

SkyGuard AI implements an **authoritative backend / presentation-only frontend** architecture. Every anomaly score, confidence metric, fault classification, and lifecycle event is calculated on the server. The user interface does not compute or fabricate synthetic numbers.

```text
       ┌───────────────────────────────┐        ┌──────────────────────────────┐
       │   Physical IoT AWS Testbed    │        │  Official IMD AWS Telemetry  │
       │   (ESP32 + Bosch BME280)      │        │  (Vijayawada, Vizag, Tirupati│
       └──────────────┬────────────────┘        └──────────────┬───────────────┘
                      │                                        │
                      ▼                                        ▼
       ┌───────────────────────────────────────────────────────────────────────┐
       │             SkyGuard Unified Telemetry Ingestion Layer                │
       │     (Schema Validation, Provenance Tagging & Timestamp UTC Sync)      │
       └──────────────────────────────────┬────────────────────────────────────┘
                                          │
                                          ▼
       ┌───────────────────────────────────────────────────────────────────────┐
       │                15-Stage Canonical Detection Pipeline                  │
       │                                                                       │
       │  1. Ingestion Normalization      8. Machine Learning Novelty (LOF)    │
       │  2. Timestamp Validation         9. Multivariate Thermodynamic Decoup │
       │  3. Historical Hydration        10. Spatial Consensus Corroboration   │
       │  4. Deterministic Overrides     11. Macro Weather Context Analysis    │
       │  5. Diurnal Baseline Drift      12. 7-Channel Evidence Fusion Engine  │
       │  6. Temporal Rate Limits        13. 5-Question Structured Explanation │
       │  7. Frozen Sensor Detection     14. Multi-Signal Station Health       │
       │                                 15. Incident Deduplication & Database │
       └──────────────────────────────────┬────────────────────────────────────┘
                                          │
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
       ┌──────────────────────────────┐       ┌────────────────────────────────┐
       │   Persistent Database Layer  │       │     Interactive Operator UI    │
       │  (Supabase PostgreSQL /      │       │   (TanStack Start + Tailwind + │
       │   Local SQLite Backing)      │       │    Recharts + Leaflet Geospatial)
       └──────────────────────────────┘       └────────────────────────────────┘
```

---

### 3. 7-Channel Evidence Fusion Framework

Rather than relying on an opaque single-score classifier, SkyGuard AI aggregates evidence across seven distinct, scientifically motivated channels. Each channel computes an independent score \([0.0, 1.0]\) alongside structured diagnostic reasoning.

| Channel #     | Channel Name                    | Analytical Basis                                           | Key Evaluated Metrics                                                                                                                                                                               |
| :------------ | :------------------------------ | :--------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Channel 1** | **Physical Data Quality**       | World Meteorological Organization (WMO) physical envelopes | Atmospheric temperature (-50°C to +65°C), Pressure (700 to 1150 hPa), Humidity (0% to 100%), non-null payload. Deterministic overrides immediately flag violations.                                 |
| **Channel 2** | **Temporal Consistency**        | Rate-of-change dynamic physical tolerances                 | Temperature delta exceeding 6.0°C/step (or >8.0°C/15min) triggers temporal spike alert; sudden drops or steps beyond thermal inertia.                                                               |
| **Channel 3** | **Statistical Baseline**        | Rolling diurnal statistics (24h/120-sample rolling window) | Robust Z-score departing \(\pm 3\sigma\) from diurnal expectation; monotonic drift accumulator (\(>0.2^\circ\text{C}/\text{hr}\)); zero-variance frozen detector.                                   |
| **Channel 4** | **Machine Learning Novelty**    | Unsupervised density estimation on local manifold          | **Local Outlier Factor (LOF v2.0.0)** trained strictly on clean baseline telemetry using `BME280_FEATURES_v2` schema. Calibrated threshold: **0.40**.                                               |
| **Channel 5** | **Multivariate Thermodynamics** | Joint psychrometric and atmospheric relations              | Temperature-humidity covariance, dewpoint depression spread, and barometric vapor pressure coupling.                                                                                                |
| **Channel 6** | **Spatial Corroboration**       | Regional 100 km radius reference network                   | Three explicit states: `SPATIAL_CORROBORATED` (neighbor consensus), `SPATIAL_NOT_CORROBORATED` (isolated divergence), `SPATIAL_EVIDENCE_UNAVAILABLE` (neutral weight reallocation without penalty). |
| **Channel 7** | **Macro Weather Context**       | Synoptic numerical weather model alignment                 | Open-Meteo context integration. Distinguishes synchronized regional front drops from solitary electrical transducer faults.                                                                         |

#### Five-Question Structured Operator Explanations

For every processed observation, the pipeline generates a structured, machine-readable explanation:

1. **What happened?**: Summary of the physical parameter movement.
2. **Why was it flagged?**: Primary deterministic, statistical, or machine learning triggers.
3. **Real Weather Event vs Sensor Anomaly**: Objective classification (`LIKELY_WEATHER_EVENT` vs `LIKELY_SENSOR_ANOMALY` vs `INSUFFICIENT_EVIDENCE`) based on thermodynamic psychrometric coherence.
4. **Recommended Operator Action**: Actionable guidance (e.g., "Inspect ADC calibration and replace sensor transducer" vs "No hardware maintenance required; regional thunderstorm in progress").
5. **Data Provenance & Hardware Health**: Explicit data source origin and updated station multi-signal health index.

---

### 4. Authoritative Machine Learning Model & Comparative Benchmark

#### Benchmark Methodology

To guarantee scientific integrity and prevent data snooping:

- **Zero-Leakage Partitioning**:
  - **Clean Baseline Training**: 576 rows of certified normal ambient telemetry (`ground_truth == 'NORMAL'`). No future records, synthetic faults, or storm events leak into training.
  - **Validation Set**: 3,070 rows across 5 stations with synthetic and real weather excursions. All detector thresholds were calibrated exclusively on validation data and permanently frozen.
  - **Test Set**: 3,135 rows across 5 stations used exclusively for out-of-sample final evaluation.

#### 4-Model Comparative Evaluation Matrix

| Evaluated Model                       | Selection Score |  ROC-AUC   | Balanced Accuracy |  Recall   | False Positive Rate | False Alarms / Station-Day | Inference Latency |
| :------------------------------------ | :-------------: | :--------: | :---------------: | :-------: | :-----------------: | :------------------------: | :---------------: |
| **Local Outlier Factor (LOF v2.0.0)** |   **0.5938**    | **0.8581** |    **0.7889**     | **90.0%** |      **32.2%**      |          **7.73**          |   **23.22 ms**    |
| Robust Covariance (Elliptic Envelope) |     0.5699      |   0.8142   |      0.4992       |   10.0%   |        98.2%        |           23.56            |     37.27 ms      |
| One-Class SVM (OCSVM)                 |     0.4962      |   0.8142   |      0.5951       |   90.0%   |        71.0%        |           17.04            |     37.27 ms      |
| Isolation Forest (iForest)            |     0.3609      |   0.7452   |      0.6346       |   32.0%   |        5.1%         |            1.22            |     233.22 ms     |

#### Scientific Selection Rationale

**Local Outlier Factor (LOF v2.0.0)** achieved the highest Selection Score (**0.5938**), the highest ROC-AUC (**0.8581**), the highest Balanced Accuracy (**0.7889**), and the highest Recall (**90.0%**), while maintaining real-time inference latency of **23.22 ms**. Unlike global density estimators (Isolation Forest), LOF’s local density estimation excels at detecting subtle contextual departures along the nonlinear diurnal curve of temperature, humidity, and barometric pressure.

---

### 5. Multi-Signal Station Health Governance

SkyGuard AI enforces the foundational operational rule: **"One anomaly != bad station"**.

A single isolated weather event or solitary spike must never mark an operational station as degraded. The `SensorHealthTracker` subsystem tracks station health on a continuous scale of 0 to 100 based on rolling multi-signal indicators:

- **Rolling anomaly frequency** (measured across rolling 20-observation windows)
- **Data packet completeness & transmission dropouts**
- **Zero-variance sensor freeze persistence**
- **Diurnal baseline drift accumulation**
- **Physical bounds & ADC sanity**

#### Canonical Station Health States

1. **`HEALTHY`** (\(\text{Score} \ge 80.0\)): All physical parameters calibrated and transmitting within normal tolerances. A solitary anomaly incurs only a minor transient penalty (~6 pts), keeping health \(\ge 85.0\).
2. **`DEGRADED`** (\(50.0 \le \text{Score} < 80.0\)): Recurring telemetry departures (\(\ge 15\%\) anomaly frequency) or accumulating baseline drift.
3. **`UNHEALTHY`** (\(\text{Score} < 50.0\)): Severe persistent hardware faults or \(\ge 30\%\) anomaly frequency. Field technician dispatch required.
4. **`OFFLINE`**: Telemetry communication gap or complete packet loss.
5. **`WARMUP`**: Initialization phase accumulating 24-hour diurnal baseline history (< 24 historical samples).
6. **`UNKNOWN / INSUFFICIENT_DATA`**: Station newly registered with zero recorded observations.

---

### 6. Operational Anomaly Lifecycle Management

Anomalies in SkyGuard AI follow a formal four-stage operational state machine backed by persistent database storage:

```text
    ┌────────────┐        Operator Ack        ┌────────────────┐
    │  DETECTED  │ ─────────────────────────> │  ACKNOWLEDGED  │
    └─────┬──────┘                            └───────┬────────┘
          │                                           │
          │ Auto-Resolve                              │ Technician Dispatched
          │ (2 consecutive normal readings)           │
          ▼                                           ▼
    ┌────────────┐     Maintenance Complete   ┌────────────────┐
    │  RESOLVED  │ <───────────────────────── │ INVESTIGATING  │
    └────────────┘                            └────────────────┘
```

#### Incident Deduplication

When consecutive observations of the same fault type occur on a station, the system **does not spawn duplicate alert notifications**. Instead, it deduplicates against the active incident, incrementing its `occurrences` count, updating the `last_detected` timestamp, and dynamically tracking duration.

---

### 7. Hardware-in-the-Loop (HIL) Testbed Architecture

SkyGuard AI was physically validated using custom IoT hardware nodes mimicking operational IMD Automatic Weather Stations:

```text
       ┌────────────────────────────────────────────────────────┐
       │              IoT Edge AWS Hardware Node                │
       │                                                        │
       │    ┌──────────────────┐        ┌──────────────────┐    │
       │    │  Bosch BME280    │ ──I2C─>│    ESP-32 MCU    │    │
       │    │  (Temp, Hum, Bar)│        │   (FreeRTOS C++) │    │
       │    └──────────────────┘        └────────┬─────────┘    │
       │                                         │              │
       │                                     WiFi / 4G          │
       │                                         │              │
       └─────────────────────────────────────────┼──────────────┘
                                                 ▼
                              POST /api/telemetry (JSON Payload)
```

- **Physical Sensor**: Bosch Sensortec BME280 digital transducer measuring temperature (\(\pm 0.5^\circ\text{C}\)), relative humidity (\(\pm 3\%\)), and barometric pressure (\(\pm 1\,\text{hPa}\)).
- **Microcontroller**: ESP32 Dual-Core Tensilica Xtensa 32-bit LX6 running modular FreeRTOS firmware.
- **Physical Fault Injection Modes**: Controlled hardware testbed support for heating/cooling excursions, moisture shielding, signal disconnection, and power-cycling transients.

---

### 8. Data Provenance & Scientific Integrity Standards

To preserve complete transparency during SIH evaluations, all telemetry records and API responses are strictly tagged with immutable provenance labels:

- **`IMD_AWS`**: Real live telemetry ingested directly from operational IMD reference stations.
- **`DEMO_SIMULATION`**: Synthetic demo dataset replay for bench evaluations and fault injections.
- **`OPEN_METEO_CONTEXT`**: External synoptic weather forecast context.
- **`OFFLINE_PREVIEW`**: Cached local inspection mode.

_Strict Guarantee_: Simulated stations (e.g., `AWS-101`) are never mislabeled as `IMD_AWS`.

---

### 9. Smart India Hackathon (SIH) Demonstration Runbook

#### 1. Quick Startup Commands

```bash
# Terminal 1: Launch Backend FastApi Service
cd Backend
python -m uvicorn ml_api_server:app --host 0.0.0.0 --port 8787 --reload

# Terminal 2: Launch Frontend Next/TanStack Service
npm run dev
```

#### 2. Demonstration Journey for Judges

1. **Overview Dashboard (`/`)**:
   - Inspect live India AWS map showing real IMD stations (Vijayawada, Vizag, Tirupati) and regional network.
   - Observe real-time KPI metrics and active alert cards.
   - Inspect the **Operational Anomaly History & Lifecycle Management** table. Click "Ack", "Investigate", or "Resolve" to demonstrate real-time state transitions.
2. **Detailed Diagnostics (`/diagnostics`)**:
   - Inspect the **7-Channel Evidence Waterfall** showing the breakdown across Physical, Temporal, Statistical, ML, Multivariate, Spatial, and Weather Context channels.
   - Inspect the **5-Question Operator Explanation Card** answering What, Why, Weather vs Sensor classification, Recommended Action, and Provenance.
   - Inspect the **LOF v2.0.0 Telemetry Card** displaying the calibrated threshold (**0.40**).
3. **ML Model Governance & Comparative Benchmarking (`/admin`)**:
   - View the 4-model comparative matrix comparing LOF v2.0.0 against Isolation Forest, OneClassSVM, and Robust Covariance.
   - Demonstrate the certified winning status of LOF (Selection Score: 0.5938, ROC-AUC: 0.8581, Recall: 90%).
   - Review the strict zero-leakage training partition disclaimers.
4. **Station Fleet Health (`/sensors`)**:
   - Demonstrate multi-signal health tracking across `HEALTHY`, `DEGRADED`, `UNHEALTHY`, `OFFLINE`, and `WARMUP`.
   - Show that solitary weather events do not falsely degrade station health ("One anomaly != bad station").
