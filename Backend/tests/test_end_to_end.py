import pandas as pd

from skyguard_ml import SkyGuard, TrainingConfig


def make_training_data(rows=80):
    timestamps = pd.date_range(
        "2026-01-01",
        periods=rows,
        freq="30min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "station_id": ["S1"] * rows,
            "timestamp": timestamps,
            "temperature": [
                30.0 + (index % 8) * 0.2
                for index in range(rows)
            ],
            "humidity": [60.0] * rows,
            "pressure": [1008.0] * rows,
            "rainfall": [0.0] * rows,
            "wind_speed": [5.0] * rows,
            "wind_direction": [180.0] * rows,
            "latitude": [16.5] * rows,
            "longitude": [80.6] * rows,
        }
    )


def test_training_and_prediction(tmp_path):
    data = make_training_data()
    artifact_dir = tmp_path / "artifacts"

    engine = SkyGuard(artifact_dir)

    report = engine.train(
        data,
        TrainingConfig(
            min_training_rows=30,
            synthetic_faults_per_type=2,
        ),
    )

    assert report["selected_model"]

    observation = data.iloc[-1].to_dict()
    history = data.iloc[:-1].to_dict(orient="records")

    result = engine.predict(
        observation=observation,
        historical_context=history,
    )

    assert "status" in result
    assert "anomaly_score" in result
    assert "sensor_health" in result
    assert "reason_codes" in result
