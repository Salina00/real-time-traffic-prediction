"""
Unit tests for data validation, cleaning, future target generation,
and chronological train/test split.
"""

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import pytest

from src.data_utils import (
    validate_traffic_record,
    clean_and_prepare_dataset,
    create_future_target
)


def test_validate_traffic_record_valid():
    valid_rec = {
        "timestamp": "2026-10-22T08:00:00Z",
        "road_id": "pune_central",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "current_speed": 35.0,
        "free_flow_speed": 50.0,
        "road_closure": False,
        "confidence": 0.95,
        "data_source": "test"
    }
    is_valid, err = validate_traffic_record(valid_rec)
    assert is_valid is True
    assert err is None


def test_validate_traffic_record_missing_fields():
    # Missing road_id
    rec = {
        "timestamp": "2026-10-22T08:00:00Z",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "current_speed": 35.0,
        "free_flow_speed": 50.0
    }
    is_valid, err = validate_traffic_record(rec)
    assert is_valid is False
    assert "road_id" in err


def test_validate_traffic_record_malformed():
    # Invalid timestamp
    rec = {
        "timestamp": "invalid-date-string",
        "road_id": "swargate",
        "latitude": 18.5,
        "longitude": 73.8,
        "current_speed": 20.0,
        "free_flow_speed": 40.0
    }
    is_valid, err = validate_traffic_record(rec)
    assert is_valid is False
    assert "ISO 8601" in err

    # Negative current speed
    rec["timestamp"] = "2026-10-22T08:00:00Z"
    rec["current_speed"] = -10.0
    is_valid, err = validate_traffic_record(rec)
    assert is_valid is False
    assert "negative" in err


def test_clean_and_prepare_dataset_duplicates():
    # Dataset with duplicates on (road_id, timestamp)
    base_t = "2026-10-22T08:00:00Z"
    rows = [
        {"timestamp": base_t, "road_id": "r1", "latitude": 18.5, "longitude": 73.8, "current_speed": 20.0, "free_flow_speed": 40.0},
        {"timestamp": base_t, "road_id": "r1", "latitude": 18.5, "longitude": 73.8, "current_speed": 20.0, "free_flow_speed": 40.0},  # duplicate
        {"timestamp": "2026-10-22T08:05:00Z", "road_id": "r1", "latitude": 18.5, "longitude": 73.8, "current_speed": 25.0, "free_flow_speed": 40.0},
    ]
    df = pd.DataFrame(rows)
    cleaned = clean_and_prepare_dataset(df)
    assert len(cleaned) == 2


def test_future_target_generation_tolerance():
    # 5-min intervals: t0, t5, t10, t15, t20, t25
    t0 = datetime(2026, 10, 22, 8, 0, tzinfo=timezone.utc)
    rows = []
    # For road 'hinjawadi', speed drops at t15
    for i in range(6):
        t = t0 + timedelta(minutes=i * 5)
        spd = 50.0 if i < 3 else 15.0  # Drops to 15 at t15
        rows.append({
            "timestamp": t.isoformat(),
            "road_id": "hinjawadi",
            "latitude": 18.59,
            "longitude": 73.73,
            "current_speed": spd,
            "free_flow_speed": 60.0,
            "road_closure": False,
            "congestion_ratio": spd / 60.0,
            "congestion_level": "low" if spd > 40 else "high",
            "data_source": "test"
        })

    df = pd.DataFrame(rows)
    cleaned = clean_and_prepare_dataset(df)

    # Future target horizon: 15 mins, tolerance: ±5 mins (10 to 20 mins)
    labeled = create_future_target(cleaned, horizon_minutes=15, tolerance_minutes=5)

    assert not labeled.empty
    # Row 0 at t=8:00 should match row 3 at t=8:15 (15 min difference)
    first_sample = labeled.iloc[0]
    assert first_sample["timestamp"] == cleaned.iloc[0]["timestamp"]
    assert first_sample["target_timestamp"] == cleaned.iloc[3]["timestamp"]
    assert first_sample["target_speed"] == 15.0
    assert first_sample["target_congestion_level"] == "high"


def test_chronological_train_test_split():
    # Test that chronological splitting keeps past in train and future in test
    t0 = datetime(2026, 10, 22, 8, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(100):
        t = t0 + timedelta(minutes=i * 5)
        rows.append({
            "timestamp": t.isoformat(),
            "road_id": "r1",
            "latitude": 18.5,
            "longitude": 73.8,
            "current_speed": 40.0,
            "free_flow_speed": 50.0,
            "road_closure": False
        })
    df = clean_and_prepare_dataset(pd.DataFrame(rows))
    labeled = create_future_target(df, horizon_minutes=15, tolerance_minutes=5)

    split_idx = int(len(labeled) * 0.8)
    train_df = labeled.iloc[:split_idx]
    test_df = labeled.iloc[split_idx:]

    max_train_time = train_df["timestamp"].max()
    min_test_time = test_df["timestamp"].min()

    # Verify no chronological overlap / no future leakage
    assert max_train_time <= min_test_time
