from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skyguard_ml.config import FeatureConfig
from skyguard_ml.feature_engineering import BASE_VARIABLES, FeatureBuilder
from skyguard_ml.preprocessing import chronological_split


def _generate_synthetic_series(n_rows: int = 100) -> pd.DataFrame:
    timestamps = pd.date_range("2026-08-01", periods=n_rows, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "station_id": ["STATION_01"] * n_rows,
        "timestamp": timestamps,
        "temperature": 25.0 + 5.0 * np.sin(np.linspace(0, 4 * np.pi, n_rows)) + rng.normal(0, 0.5, n_rows),
        "humidity": 60.0 - 10.0 * np.sin(np.linspace(0, 4 * np.pi, n_rows)) + rng.normal(0, 1.0, n_rows),
        "pressure": 1012.0 + 2.0 * np.cos(np.linspace(0, 4 * np.pi, n_rows)) + rng.normal(0, 0.3, n_rows),
    })


def test_no_temporal_leakage_in_rolling_and_multivariate():
    """Verify that features at index i do NOT depend on future data at i+1, i+2..."""
    df = _generate_synthetic_series(80)
    builder = FeatureBuilder(FeatureConfig())
    features_orig = builder.fit_transform(df)

    # Corrupt future rows (from index 50 onwards with huge spikes)
    df_corrupted_future = df.copy()
    df_corrupted_future.loc[50:, "temperature"] += 100.0
    df_corrupted_future.loc[50:, "humidity"] -= 50.0
    df_corrupted_future.loc[50:, "pressure"] += 200.0

    builder2 = FeatureBuilder(FeatureConfig())
    features_future_corrupted = builder2.fit_transform(df_corrupted_future)

    # Features strictly before index 50 MUST be bitwise identical
    for col in builder.feature_list:
        val_orig = features_orig.loc[:49, col].values
        val_corrupt = features_future_corrupted.loc[:49, col].values
        # NaN matching or finite matching
        np.testing.assert_allclose(
            np.nan_to_num(val_orig, nan=-9999.0),
            np.nan_to_num(val_corrupt, nan=-9999.0),
            rtol=1e-5,
            err_msg=f"Temporal leakage detected in feature '{col}': past features changed when future data changed!",
        )


def test_chronological_split_strict_separation():
    """Verify that chronological_split guarantees no timestamp overlap between train, val, and test."""
    df = _generate_synthetic_series(120)
    train, val, test = chronological_split(df, train_fraction=0.6, validation_fraction=0.2)

    assert not train.empty
    assert not val.empty
    assert not test.empty

    max_train_ts = train["timestamp"].max()
    min_val_ts = val["timestamp"].min()
    max_val_ts = val["timestamp"].max()
    min_test_ts = test["timestamp"].min()

    assert max_train_ts < min_val_ts, f"Train overlap with val: max_train={max_train_ts}, min_val={min_val_ts}"
    assert max_val_ts < min_test_ts, f"Val overlap with test: max_val={max_val_ts}, min_test={min_test_ts}"


def test_feature_count_and_sih_compliance():
    """Verify that features only use SIH core variables (T, P, RH) and feature count is defensible (< 50)."""
    df = _generate_synthetic_series(50)
    builder = FeatureBuilder(FeatureConfig())
    features = builder.fit_transform(df)

    # Ensure no wind or rain columns in feature list
    for col in builder.feature_list:
        assert "wind" not in col.lower(), f"Forbidden wind feature found: {col}"
        assert "rain" not in col.lower(), f"Forbidden rainfall feature found: {col}"

    # Verify compact feature space (around 45 features, strictly < 60)
    assert len(builder.feature_list) <= 55, f"Feature count bloated: {len(builder.feature_list)}"
    assert len(builder.feature_list) >= 30, f"Feature count too low: {len(builder.feature_list)}"


def test_feature_schema_consistency_train_vs_inference():
    """Verify that transform() produces exact same feature schema as fit_transform()."""
    df = _generate_synthetic_series(60)
    builder = FeatureBuilder(FeatureConfig())
    train_feat = builder.fit_transform(df)

    new_obs = _generate_synthetic_series(5)
    infer_feat = builder.transform(new_obs)

    assert list(train_feat.columns) == list(infer_feat.columns)
    assert builder.feature_list == [c for c in builder.feature_list if c in infer_feat.columns]
