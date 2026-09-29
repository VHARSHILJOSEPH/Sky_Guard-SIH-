from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skyguard_ml.config import TrainingConfig
from skyguard_ml.model_comparison import run_model_comparison


def _create_test_dataset(n_hours: int = 150) -> list[dict]:
    rng = np.random.default_rng(42)
    records = []
    base_time = pd.Timestamp("2026-08-01T00:00:00Z")

    for stn in ["STATION_A", "STATION_B"]:
        for h in range(n_hours):
            t = base_time + pd.Timedelta(hours=h)
            records.append({
                "station_id": stn,
                "timestamp": t.isoformat(),
                "temperature": float(28.0 + 5.0 * np.sin(h * np.pi / 12) + rng.normal(0, 0.4)),
                "humidity": float(np.clip(65.0 - 15.0 * np.sin(h * np.pi / 12) + rng.normal(0, 1.0), 20.0, 95.0)),
                "pressure": float(1010.0 + 1.5 * np.cos(h * np.pi / 6) + rng.normal(0, 0.3)),
            })
    return records


def test_model_comparison_execution(tmp_path):
    """Run model comparison end-to-end and assert valid comparative metrics."""
    data = _create_test_dataset(120)
    config = TrainingConfig(
        synthetic_faults_per_type=3,
        min_training_rows=30,
        random_state=42,
    )

    result = run_model_comparison(data, config=config, output_dir=tmp_path)

    assert "winning_model" in result
    assert result["winning_model"] in ["IsolationForest", "LOF"]
    assert len(result["comparison_table"]) == 4

    # Verify candidate models were all evaluated
    evaluated_models = [r["Model"] for r in result["comparison_table"]]
    assert set(evaluated_models) == {"IsolationForest", "LOF", "OneClassSVM", "RobustCovariance"}

    # Winning model must meet strict SIH operational requirements
    winner_row = next(r for r in result["comparison_table"] if r["Model"] == result["winning_model"])
    assert winner_row["FPR"] < 0.15, f"Winning model {result['winning_model']} has excessive FPR: {winner_row['FPR']}"
    assert winner_row["Specificity"] > 0.85, f"Winning model {result['winning_model']} has poor specificity: {winner_row['Specificity']}"
    assert winner_row["Balanced Acc"] > 0.70, f"Winning model {result['winning_model']} balanced accuracy too low: {winner_row['Balanced Acc']}"
    assert winner_row["ROC-AUC"] > 0.75, f"Winning model {result['winning_model']} ROC-AUC too low: {winner_row['ROC-AUC']}"

    # Verify report files were created
    assert (tmp_path / "model_comparison_report.json").exists()
    assert (tmp_path / "model_comparison_report.md").exists()
