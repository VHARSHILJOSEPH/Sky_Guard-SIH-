"""
SkyGuard AI — Incident Lifecycle & Alert Deduplication Manager.

Handles:
1. Alert deduplication: grouping consecutive observations of the same fault
   into a single persistent incident with occurrence count and duration.
2. Incident lifecycle tracking: DETECTED -> ACKNOWLEDGED -> INVESTIGATING -> RESOLVED.
3. Thread-safe in-memory cache with persistent backing support.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import threading
from typing import Any, Optional

from .canonical_schema import AnomalyIncidentModel, IncidentLifecycleState, SeverityLevel


class IncidentManager:
    """
    Manages active and historical anomaly incidents, deduplicating alerts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Active incidents keyed by (session_id, station_id, anomaly_type)
        self._active_incidents: dict[tuple[Optional[str], str, str], dict[str, Any]] = {}
        # Consecutively normal readings count since last anomaly
        self._normal_counter: dict[tuple[Optional[str], str], int] = {}
        # Completed historical incidents
        self._incident_history: list[dict[str, Any]] = []

    def _generate_incident_id(self, station_id: str, anomaly_type: str, timestamp_str: str) -> str:
        """Create a deterministic, readable incident ID."""
        ts_clean = timestamp_str.replace(":", "").replace("-", "").replace("T", "_").replace("Z", "")
        h = hashlib.md5(f"{station_id}_{anomaly_type}_{timestamp_str}".encode()).hexdigest()[:6]
        return f"INC_{station_id}_{anomaly_type}_{ts_clean[:13]}_{h}"

    def process_observation_incident(
        self,
        station_id: str,
        timestamp_str: str,
        is_anomaly: bool,
        anomaly_type: str,
        severity: SeverityLevel,
        session_id: Optional[str] = None,
    ) -> Optional[AnomalyIncidentModel]:
        """
        Update or create an incident based on the pipeline decision.
        """
        with self._lock:
            key = (session_id, station_id, anomaly_type)
            stn_key = (session_id, station_id)
            current_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            if is_anomaly and anomaly_type not in ("NONE", "NORMAL", "UNKNOWN_ANOMALY"):
                self._normal_counter[stn_key] = 0

                # 1. Deduplication: Check if there is an existing active incident for this fault
                if key in self._active_incidents:
                    inc = self._active_incidents[key]
                    first_dt = datetime.fromisoformat(inc["first_detected"].replace("Z", "+00:00"))
                    inc["last_detected"] = timestamp_str
                    inc["occurrence_count"] += 1
                    inc["duration_seconds"] = max(0.0, (current_dt - first_dt).total_seconds())

                    # Escalate severity if incoming reading is higher
                    sev_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
                    if sev_rank.get(severity, 1) > sev_rank.get(inc["severity"], 1):
                        inc["severity"] = severity

                    return AnomalyIncidentModel(**inc)

                # 2. New Incident Creation
                incident_id = self._generate_incident_id(station_id, anomaly_type, timestamp_str)
                inc_data: dict[str, Any] = {
                    "incident_id": incident_id,
                    "status": "DETECTED",
                    "first_detected": timestamp_str,
                    "last_detected": timestamp_str,
                    "occurrence_count": 1,
                    "duration_seconds": 0.0,
                    "anomaly_type": anomaly_type,
                    "severity": severity,
                }
                self._active_incidents[key] = inc_data
                return AnomalyIncidentModel(**inc_data)

            else:
                # Normal reading: increment consecutive normal counter
                self._normal_counter[stn_key] = self._normal_counter.get(stn_key, 0) + 1

                # If 2 consecutive normal readings occur, resolve any active incidents for this station
                if self._normal_counter[stn_key] >= 2:
                    keys_to_resolve = [k for k in self._active_incidents if k[0] == session_id and k[1] == station_id]
                    for r_key in keys_to_resolve:
                        inc = self._active_incidents.pop(r_key)
                        inc["status"] = "RESOLVED"
                        self._incident_history.append(inc)

                return None

    def update_lifecycle_status(
        self,
        incident_id: str,
        new_status: IncidentLifecycleState,
    ) -> Optional[dict[str, Any]]:
        """Update lifecycle status of an active or historical incident."""
        with self._lock:
            # Check active
            for key, inc in self._active_incidents.items():
                if inc["incident_id"] == incident_id:
                    inc["status"] = new_status
                    if new_status == "RESOLVED":
                        self._active_incidents.pop(key)
                        self._incident_history.append(inc)
                    return dict(inc)

            # Check history
            for inc in self._incident_history:
                if inc["incident_id"] == incident_id:
                    inc["status"] = new_status
                    return dict(inc)

            return None

    def list_incidents(
        self,
        status: Optional[IncidentLifecycleState] = None,
        station_id: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve recent incidents filtered by status or station."""
        with self._lock:
            combined = list(self._active_incidents.values()) + list(reversed(self._incident_history))
            if status:
                combined = [i for i in combined if i["status"] == status]
            if station_id:
                combined = [i for i in combined if i.get("incident_id", "").startswith(f"INC_{station_id}")]
            return combined[:limit]


# Global singleton instance
incident_manager = IncidentManager()
