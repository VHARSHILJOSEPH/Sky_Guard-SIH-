from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np


@dataclass
class SensorHealthTracker:
    """
    Authoritative Station Health Subsystem.
    
    Derives station health from telemetry history and multi-channel evidence:
    - Telemetry freshness and communication gaps
    - Missing data frequency
    - Frozen sensor readings (zero variance)
    - Standardized baseline drift accumulation
    - Physical bounds / data-quality failures
    - Anomaly frequency over rolling observation window
    - Parameter consistency
    
    Strictly adheres to the rule: "One anomaly != bad station".
    Exposes canonical states:
    - HEALTHY
    - DEGRADED
    - UNHEALTHY
    - OFFLINE
    - WARMUP
    - UNKNOWN / INSUFFICIENT_DATA
    """

    history: dict[str, list[float]] = field(default_factory=dict)
    anomaly_history: dict[str, list[bool]] = field(default_factory=dict)
    issue_history: dict[str, list[str]] = field(default_factory=dict)
    last_timestamps: dict[str, str] = field(default_factory=dict)
    max_history: int = 100

    def update(
        self,
        station_id: str,
        anomaly_score: float,
        anomaly_type: str,
        missing_score: float = 0.0,
        drift_score: float = 0.0,
        spatial_disagreement: Optional[float] = None,
        forecast_score: Optional[float] = None,
        is_warmup: bool = False,
        is_offline: bool = False,
        is_frozen: bool = False,
        quality_score: float = 1.0,
        history_len: int = 0,
        timestamp: Optional[str] = None,
        severity: str = "LOW",
    ) -> dict[str, Any]:
        values = self.history.setdefault(station_id, [])
        anom_flags = self.anomaly_history.setdefault(station_id, [])
        issues = self.issue_history.setdefault(station_id, [])

        if timestamp:
            self.last_timestamps[station_id] = timestamp

        is_anom = anomaly_type not in ("NORMAL", "NONE", "WARMUP") and anomaly_score >= 0.40
        anom_flags.append(is_anom)
        if len(anom_flags) > self.max_history:
            del anom_flags[:-self.max_history]

        if is_anom and anomaly_type not in ("NORMAL", "NONE", "WARMUP"):
            issue_label = anomaly_type.replace("_", " ").title()
            if issue_label not in issues:
                issues.append(issue_label)
            if len(issues) > 10:
                del issues[:-10]

        # Calculate rolling anomaly frequency (last 20 observations)
        window = anom_flags[-20:] if len(anom_flags) >= 20 else anom_flags
        anomaly_freq = (sum(window) / len(window)) if window else 0.0

        # Multi-signal penalty computation:
        # A single isolated anomaly creates only a minor transient penalty (~6 pts),
        # preserving the rule that one anomaly does not designate an unhealthy station.
        single_anomaly_penalty = 6.0 if is_anom else 0.0
        frequency_penalty = 30.0 * anomaly_freq
        missing_penalty = 25.0 * float(missing_score)
        drift_penalty = 20.0 * float(drift_score)
        frozen_penalty = 20.0 if (is_frozen or anomaly_type == "FROZEN_SENSOR") else 0.0
        quality_penalty = 25.0 * max(0.0, 1.0 - float(quality_score))
        spatial_penalty = 10.0 * (spatial_disagreement or 0.0)

        total_penalty = (
            single_anomaly_penalty
            + frequency_penalty
            + missing_penalty
            + drift_penalty
            + frozen_penalty
            + quality_penalty
            + spatial_penalty
        )

        raw_score = float(np.clip(100.0 - total_penalty, 0.0, 100.0))
        values.append(raw_score)
        if len(values) > self.max_history:
            del values[:-self.max_history]

        # Trend analysis (last 5 vs preceding 5)
        recent = values[-5:]
        older = values[-10:-5]
        trend = "STABLE"
        if len(older) >= 2 and len(recent) >= 2:
            recent_mean = float(np.mean(recent))
            older_mean = float(np.mean(older))
            if recent_mean < older_mean - 4.0:
                trend = "DECLINING"
            elif recent_mean > older_mean + 4.0:
                trend = "IMPROVING"

        # Categorize Canonical Status
        contributors: list[str] = []
        if is_offline or anomaly_type == "COMMUNICATION_FAILURE":
            status = "OFFLINE"
            primary_reason = "Telemetry communication gap: station transmission interrupted or missing packet"
            contributors.append("Complete communication packet loss")
        elif is_warmup or (history_len > 0 and history_len < 24 and anomaly_type in ("NORMAL", "NONE", "WARMUP")):
            status = "WARMUP"
            primary_reason = f"Initialization phase: accumulating 24-hour diurnal history ({history_len}/24 samples)"
            contributors.append(f"Insufficient baseline history ({history_len} of 24 observations)")
        elif raw_score < 50.0 or (severity == "CRITICAL" and anomaly_type in ("DATA_QUALITY_ANOMALY", "CORRUPTED_DATA") and anomaly_freq >= 0.2):
            status = "UNHEALTHY"
            primary_reason = "Critical hardware degradation or persistent data corruption requiring field maintenance"
        elif raw_score < 80.0 or anomaly_freq >= 0.15 or drift_score >= 0.4:
            status = "DEGRADED"
            primary_reason = "Operational sensor degradation: recurring telemetry departures or transducer drift"
        elif len(values) == 0 and history_len == 0:
            status = "UNKNOWN / INSUFFICIENT_DATA"
            primary_reason = "No telemetry history available to assess station health"
        else:
            status = "HEALTHY"
            primary_reason = "Nominal operation: all sensor channels calibrated and transmitting within tolerances"

        # Build contributors evidence trail
        if anomaly_freq >= 0.2:
            contributors.append(f"Elevated anomaly frequency ({anomaly_freq * 100:.0f}% of recent observations)")
        elif is_anom:
            contributors.append("Single isolated telemetry excursion under monitoring (nominal station health maintained)")
        if missing_score >= 0.25:
            contributors.append(f"Missing data packets ({missing_score * 100:.0f}% field drop rate)")
        if drift_score >= 0.35:
            contributors.append(f"Systematic baseline drift detected ({drift_score:.2f} accumulator)")
        if is_frozen or anomaly_type == "FROZEN_SENSOR":
            contributors.append("Zero variance detected across consecutive sampling intervals (frozen transducer)")
        if quality_score < 0.8:
            contributors.append("Physical climatological bounds rejection on raw telemetry")
        if spatial_disagreement and spatial_disagreement >= 0.5:
            contributors.append("Spatial disagreement with nearby AWS reference network")

        if not contributors:
            contributors.append("All physical parameters within certified operational envelopes")

        degradation_warning = trend == "DECLINING" and raw_score < 75.0

        return {
            "score": round(raw_score, 1),
            "status": status,
            "trend": trend,
            "primary_reason": primary_reason,
            "contributors": contributors,
            "recent_issues": list(issues[-5:]),
            "degradation_warning": degradation_warning,
            "metrics": {
                "anomaly_frequency": round(anomaly_freq, 3),
                "missing_rate": round(float(missing_score), 3),
                "drift_score": round(float(drift_score), 3),
                "observations_tracked": len(values),
            },
            "message": (
                "Potential sensor degradation detected. Inspection recommended."
                if degradation_warning
                else None
            ),
        }

    def get_health_status(self, station_id: Optional[str] = None) -> dict[str, Any]:
        """Retrieve current health status for a specific station or overall mesh summary."""
        if not station_id or station_id not in self.history or len(self.history[station_id]) == 0:
            return {
                "score": 100.0,
                "status": "UNKNOWN / INSUFFICIENT_DATA",
                "trend": "STABLE",
                "primary_reason": "Insufficient telemetry history to compute multi-signal health index",
                "contributors": ["No prior historical observations in tracking window"],
                "recent_issues": [],
                "degradation_warning": False,
                "metrics": {
                    "anomaly_frequency": 0.0,
                    "missing_rate": 0.0,
                    "drift_score": 0.0,
                    "observations_tracked": 0,
                },
            }

        values = self.history[station_id]
        anom_flags = self.anomaly_history.get(station_id, [])
        issues = self.issue_history.get(station_id, [])
        window = anom_flags[-20:] if len(anom_flags) >= 20 else anom_flags
        anomaly_freq = (sum(window) / len(window)) if window else 0.0
        current_score = values[-1] if values else 100.0

        if current_score < 50.0 or anomaly_freq >= 0.3:
            status = "UNHEALTHY"
            primary_reason = "Persistent hardware anomalies or repeated data quality violations"
        elif current_score < 80.0 or anomaly_freq >= 0.15:
            status = "DEGRADED"
            primary_reason = "Operational sensor degradation or elevated anomaly frequency"
        elif len(values) < 24:
            status = "WARMUP"
            primary_reason = f"Station baseline warmup in progress ({len(values)}/24 observations)"
        else:
            status = "HEALTHY"
            primary_reason = "Nominal operation: all sensor channels calibrated and transmitting within tolerances"

        contributors = []
        if anomaly_freq >= 0.2:
            contributors.append(f"Elevated anomaly frequency ({anomaly_freq * 100:.0f}% of recent observations)")
        if not contributors:
            contributors.append("All physical parameters within certified operational envelopes")

        return {
            "score": round(current_score, 1),
            "status": status,
            "trend": "STABLE",
            "primary_reason": primary_reason,
            "contributors": contributors,
            "recent_issues": list(issues[-5:]),
            "degradation_warning": current_score < 75.0,
            "metrics": {
                "anomaly_frequency": round(anomaly_freq, 3),
                "missing_rate": 0.0,
                "drift_score": 0.0,
                "observations_tracked": len(values),
            },
        }


# Global singleton instance
sensor_health_tracker = SensorHealthTracker()
