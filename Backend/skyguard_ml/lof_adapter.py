"""
SkyGuard AI — LOF Adapter for Evidence Fusion & Pipeline Integration

Connects the standalone SkyGuardLOF (BME280 LOF model) to the existing SkyGuard
evidence-fusion and anomaly-classification pipeline.
Strictly preserves the underlying model logic: this adapter is a thin translation wrapper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from skyguard_lof import SkyGuardLOF, SkyGuardLOFConfig


class SkyGuardLOFAdapter:
    """
    Thin integration wrapper around SkyGuardLOF.
    Bridges between the standalone BME280 LOF model and the multi-source
    SkyGuard evidence-fusion architecture.
    """

    def __init__(
        self,
        artifact_path: Optional[str | Path] = None,
        config: Optional[SkyGuardLOFConfig] = None,
    ) -> None:
        if artifact_path is None:
            # Check environment or look in parent directory / artifacts directory
            env_path = os.environ.get("SKYGUARD_MODEL_PATH")
            if env_path and Path(env_path).exists():
                artifact_path = Path(env_path)
            else:
                candidates = [
                    Path("skyguard_lof_bme280_calibrated.joblib"),
                    Path("../skyguard_lof_bme280_calibrated.joblib"),
                    Path(__file__).resolve().parent.parent / "skyguard_lof_bme280_calibrated.joblib",
                    Path(__file__).resolve().parent / "artifacts" / "skyguard_lof_bme280_calibrated.joblib",
                ]
                for cand in candidates:
                    if cand.exists():
                        artifact_path = cand
                        break

        self.model: Optional[SkyGuardLOF] = None
        self.artifact_path = artifact_path
        self.config = config or SkyGuardLOFConfig()

        if self.artifact_path and Path(self.artifact_path).exists():
            try:
                self.model = SkyGuardLOF.load(self.artifact_path)
            except Exception:
                self.model = None

    def is_ready(self) -> bool:
        """Check whether the model is loaded and threshold calibrated."""
        return self.model is not None and self.model.threshold is not None

    def predict(
        self,
        station_id: str,
        observation: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """
        Run inference and return structured ML result conforming to LOF spec.
        """
        if self.model is None:
            return {
                "station_id": station_id,
                "timestamp": str(observation.get("timestamp")),
                "status": "UNAVAILABLE",
                "model": "LOF",
                "model_version": "LOF_BME280_v1.0",
                "ml_anomaly": False,
                "reason": "LOF model artifact not loaded",
                "raw_lof_score": None,
                "normalized_lof_score": None,
                "threshold": None,
            }

        return self.model.predict(
            station_id=station_id,
            observation=observation,
            history=history,
        )

    def extract_evidence_score(self, lof_result: dict[str, Any]) -> Optional[float]:
        """
        Extract normalized anomaly score (0.0 to 1.0) for evidence fusion.
        Returns None if model is in WARMUP, UNCALIBRATED, or INVALID_INPUT.
        """
        if lof_result.get("status") == "EVALUATED":
            score = lof_result.get("normalized_lof_score")
            if score is not None:
                return float(max(0.0, min(1.0, score)))
        return None
