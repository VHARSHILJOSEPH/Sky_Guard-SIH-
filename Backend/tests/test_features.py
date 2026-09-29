import pandas as pd

from skyguard_ml.baselines import BaselineStore
from skyguard_ml.config import FeatureConfig
from skyguard_ml.feature_engineering import (
    BASE_VARIABLES,
    FeatureBuilder,
)


def make_data():
    timestamps = pd.date_range(
        "2026-01-01",
        periods=20,
        freq="30min",
        tz="UTC",
    )

    return pd.DataFrame(
        {
            "station_id": ["S1"] * 20,
            "timestamp": timestamps,
            "temperature": range(20),
            "humidity": [60.0] * 20,
            "pressure": [1008.0] * 20,
            "rainfall": [0.0] * 20,
            "wind_speed": [5.0] * 20,
            "wind_direction": [180.0] * 20,
            "latitude": [16.5] * 20,
            "longitude": [80.6] * 20,
        }
    )


def test_lag_does_not_use_future_value():
    frame = make_data()

    baseline = BaselineStore(minimum_samples=2)
    baseline.fit(frame, BASE_VARIABLES)

    builder = FeatureBuilder(
        FeatureConfig(lags=(1,), rolling_windows=(3,))
    )

    features = builder.fit_transform(frame, baseline)

    first_row = features.iloc[0]

    assert pd.isna(first_row["temperature_lag_1"])
    assert pd.isna(first_row["temperature_delta"])
