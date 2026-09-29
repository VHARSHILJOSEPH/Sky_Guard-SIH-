"""Rule-based detectors (physical bounds, etc.).

Adopted from the Chatgpt ML module — a cheap, high-value safety net
that flags readings outside physically plausible ranges.
"""

from __future__ import annotations

from typing import Any


PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (-90.0, 60.0),     # °C
    "humidity": (0.0, 100.0),          # %
    "pressure": (500.0, 1100.0),       # hPa
    "wind_speed": (0.0, 50.0),         # m/s
    "rainfall": (0.0, 1000.0),         # mm
}


def check_physical_bounds(
    observation: dict[str, Any],
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Check whether an observation violates physical bounds.

    Parameters
    ----------
    observation : dict
        A single weather observation with variable keys.
    bounds : dict, optional
        Custom bounds per variable as ``{name: (lo, hi)}``.
        Falls back to ``PHYSICAL_BOUNDS`` when *None*.

    Returns
    -------
    dict
        ``score``      – 1.0 if any violation, else 0.0
        ``violations`` – list of ``{variable, value, min, max}`` dicts
        ``affected``   – first affected variable name, or *None*
    """
    bounds = bounds or PHYSICAL_BOUNDS

    violations: list[dict[str, Any]] = []
    affected: str | None = None

    for variable, (lo, hi) in bounds.items():
        value = observation.get(variable)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        if not (lo <= numeric <= hi):
            violations.append(
                {
                    "variable": variable,
                    "value": numeric,
                    "min": lo,
                    "max": hi,
                }
            )
            if affected is None:
                affected = variable

    return {
        "score": 1.0 if violations else 0.0,
        "violations": violations,
        "affected": affected,
    }
