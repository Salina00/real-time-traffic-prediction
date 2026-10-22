"""
Unit tests for Supervised Random Forest Congestion Prediction Module.
(src/congestion_prediction.py)

Uses small labeled synthetic test fixtures explicitly marked as test data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from src.congestion_prediction import (
    extract_features,
    train_and_evaluate_congestion_model,
    CongestionPredictor,
    TrafficPredictor
)
from src.data_utils import clean_and_prepare_dataset, create_future_target


def _generate_synthetic_prediction_df(num_intervals=100) -> pd.DataFrame:
    """
    Generates chronological periodic synthetic observations across 2 roads
    with ~15 min target spacing for unit testing.
    """
    now = datetime(2026, 10, 22, 10, 0, 0, tzinfo=timezone.utc)
    records = []
    for i in range(num_intervals):
        t = (now + timedelta(minutes=i * 5)).isoformat()
        for r_id, base_speed in [("pune_central", 45.0), ("swargate", 40.0)]:
            # Create varied speeds
            speed = max(5.0, base_speed - (i % 25))
            records.append({
                "timestamp": t,
                "road_id": r_id,
                "road_name": r_id.replace("_", " ").title(),
                "latitude": 18.52,
                "longitude": 73.85,
                "current_speed": speed,
                "free_flow_speed": base_speed,
                "confidence": 1.0,
                "road_closure": False,
                "data_source": "synthetic_test"
            })
    return pd.DataFrame(records)


def test_extract_features_valid():
    """Verify feature matrix construction, cyclical encoding, and lag features."""
    df = _generate_synthetic_prediction_df(20)
    cleaned = clean_and_prepare_dataset(df)
    X, encoder = extract_features(cleaned)

    expected_cols = [
        "road_encoded", "hour", "sin_hour", "cos_hour", "day_of_week",
        "is_weekend", "current_speed", "free_flow_speed", "congestion_ratio",
        "prev_speed", "confidence", "road_closure"
    ]
    for col in expected_cols:
        assert col in X.columns
    assert len(X) == len(cleaned)
    assert not X.isnull().any().any()


def test_future_target_construction_and_no_leakage():
    """Verify future target generation matches 15-min horizon without lookahead leakage."""
    df = _generate_synthetic_prediction_df(40)
    cleaned = clean_and_prepare_dataset(df)
    labeled = create_future_target(cleaned, horizon_minutes=15, tolerance_minutes=5)

    assert not labeled.empty
    for _, row in labeled.iterrows():
        t_curr = pd.to_datetime(row["timestamp"])
        t_target = pd.to_datetime(row["target_timestamp"])
        delta_min = (t_target - t_curr).total_seconds() / 60.0
        # Must be strictly in the future within (15 - 5) and (15 + 5)
        assert 10.0 <= delta_min <= 20.0


def test_insufficient_training_data_handling(tmp_path):
    """Verify that attempting to train on an undersized dataset raises ValueError."""
    small_df = _generate_synthetic_prediction_df(5)  # Only 10 total rows
    small_file = tmp_path / "small_test.csv"
    small_df.to_csv(small_file, index=False)

    with pytest.raises(ValueError, match="Insufficient observations"):
        train_and_evaluate_congestion_model(data_path=small_file)


def test_congestion_predictor_inference():
    """Verify that CongestionPredictor generates valid predictions and probabilities."""
    predictor = CongestionPredictor()
    if not predictor.is_ready():
        pytest.skip("Model artifacts not yet saved.")

    res = predictor.predict(
        road_id="pune_central",
        current_speed=20.0,
        free_flow_speed=45.0,
        road_closure=False
    )

    assert "predicted_congestion_level_15m" in res
    assert res["predicted_congestion_level_15m"] in predictor.model.classes_
    assert "probabilities" in res
    assert isinstance(res["probabilities"], dict)
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=0.05)


def test_zero_free_flow_speed_prediction():
    """Verify that predictor safely handles zero free-flow speed without division-by-zero."""
    predictor = CongestionPredictor()
    if not predictor.is_ready():
        pytest.skip("Model artifacts not yet saved.")

    res = predictor.predict(
        road_id="swargate",
        current_speed=0.0,
        free_flow_speed=0.0,  # Edge case: zero free flow
        road_closure=True
    )
    assert "predicted_congestion_level_15m" in res
    assert res["current_ratio"] == 0.0
