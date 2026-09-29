"""
SkyGuard AI — Multi-Station Scenario Orchestrator.

Constructs paired multi-station benchmarks:
- CASE A: One station becomes anomalous while nearby peer stations remain normal.
  (Spatial evidence indicates peer disagreement -> Sensor Fault).
- CASE B: Several nearby stations change coherently.
  (Spatial evidence indicates peer agreement -> Weather Event).
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .fault_generator import inject_fault_into_slice
from .weather_generator import inject_convective_storm_front


def inject_case_a_isolated_fault(
    records_by_station: dict[str, list[dict[str, Any]]],
    target_station_id: str,
    peer_station_ids: Sequence[str],
    start_index: int,
    duration_hours: int = 8,
    fault_type: str = "TEMPERATURE_DRIFT",
    severity: str = "MODERATE",
    scenario_id: str = "SCN_MULTI_CASE_A_ISOLATED_DRIFT",
    rng: np.random.Generator | None = None,
) -> None:
    """
    CASE A: Target station suffers a sensor fault, while nearby peer stations
    remain completely normal.
    """
    rng = rng or np.random.default_rng(42)

    if target_station_id not in records_by_station:
        return

    records = records_by_station[target_station_id]
    s_idx = max(0, min(start_index, len(records) - 1))
    e_idx = min(s_idx + duration_hours, len(records))

    slice_to_fault = records[s_idx:e_idx]
    faulted_slice = inject_fault_into_slice(
        clean_slice=slice_to_fault,
        fault_type=fault_type,
        severity=severity,  # type: ignore
        scenario_id=scenario_id,
        rng=rng,
        start_offset=0,
        duration=duration_hours,
    )

    records[s_idx:e_idx] = faulted_slice

    # Ensure peer stations stay marked as NORMAL but note the case reference
    for peer_id in peer_station_ids:
        if peer_id not in records_by_station:
            continue
        peer_recs = records_by_station[peer_id]
        p_s = max(0, min(start_index, len(peer_recs) - 1))
        p_e = min(p_s + duration_hours, len(peer_recs))
        for i in range(p_s, p_e):
            peer_recs[i]["description"] = f"Clean peer reference station for {scenario_id}"


def inject_case_b_coherent_event(
    records_by_station: dict[str, list[dict[str, Any]]],
    station_ids: Sequence[str],
    start_index: int,
    duration_hours: int = 5,
    scenario_id: str = "SCN_MULTI_CASE_B_COHERENT_STORM",
) -> None:
    """
    CASE B: Clustered stations all change coherently during a regional storm.
    """
    inject_convective_storm_front(
        records_by_station=records_by_station,
        station_ids=station_ids,
        start_index=start_index,
        duration_hours=duration_hours,
        scenario_id=scenario_id,
        propagation_lags={
            station_ids[0]: 0,
            station_ids[1]: 1 if len(station_ids) > 1 else 0,
            station_ids[2]: 1 if len(station_ids) > 2 else 0,
        },
    )
