"""
Unit tests for congestion ratio calculation and classification logic.
"""

import math
import pytest
from src.congestion import (
    calculate_congestion_ratio,
    classify_congestion,
    LEVEL_LOW,
    LEVEL_MODERATE,
    LEVEL_HIGH,
    LEVEL_CLOSED,
    LEVEL_UNKNOWN
)


def test_calculate_congestion_ratio_standard():
    assert calculate_congestion_ratio(40.0, 50.0) == 0.8
    assert calculate_congestion_ratio(25.0, 50.0) == 0.5
    assert calculate_congestion_ratio(10.0, 50.0) == 0.2
    assert calculate_congestion_ratio(0.0, 50.0) == 0.0


def test_calculate_congestion_ratio_invalid():
    # Division by zero or negative
    assert calculate_congestion_ratio(30.0, 0.0) is None
    assert calculate_congestion_ratio(30.0, -10.0) is None
    assert calculate_congestion_ratio(-5.0, 50.0) is None

    # Missing / None / NaN
    assert calculate_congestion_ratio(None, 50.0) is None
    assert calculate_congestion_ratio(30.0, None) is None
    assert calculate_congestion_ratio(float("nan"), 50.0) is None
    assert calculate_congestion_ratio(30.0, float("nan")) is None


def test_classify_congestion_thresholds():
    # >= 0.70 -> low
    ratio, level = classify_congestion(35.0, 50.0)  # ratio = 0.70
    assert ratio == 0.70
    assert level == LEVEL_LOW

    ratio, level = classify_congestion(45.0, 50.0)  # ratio = 0.90
    assert level == LEVEL_LOW

    # 0.40 <= ratio < 0.70 -> moderate
    ratio, level = classify_congestion(20.0, 50.0)  # ratio = 0.40
    assert ratio == 0.40
    assert level == LEVEL_MODERATE

    ratio, level = classify_congestion(34.0, 50.0)  # ratio = 0.68
    assert level == LEVEL_MODERATE

    # < 0.40 -> high
    ratio, level = classify_congestion(15.0, 50.0)  # ratio = 0.30
    assert ratio == 0.30
    assert level == LEVEL_HIGH

    ratio, level = classify_congestion(0.0, 50.0)   # ratio = 0.00
    assert level == LEVEL_HIGH


def test_road_closure_override():
    # Even if current_speed == free_flow_speed, road_closure=True must yield 'closed'
    ratio, level = classify_congestion(50.0, 50.0, road_closure=True)
    assert level == LEVEL_CLOSED

    # When speed is zero and road_closure is True
    ratio, level = classify_congestion(0.0, 50.0, road_closure=True)
    assert level == LEVEL_CLOSED

    # When free_flow_speed is 0 but road is closed
    ratio, level = classify_congestion(0.0, 0.0, road_closure=True)
    assert level == LEVEL_CLOSED


def test_missing_and_zero_free_flow_speed():
    ratio, level = classify_congestion(30.0, 0.0, road_closure=False)
    assert ratio is None
    assert level == LEVEL_UNKNOWN

    ratio, level = classify_congestion(None, 50.0, road_closure=False)
    assert ratio is None
    assert level == LEVEL_UNKNOWN


def test_custom_thresholds():
    custom = {"low_min": 0.85, "moderate_min": 0.55}
    # 0.80 would normally be low, but under custom it is moderate
    _, level = classify_congestion(40.0, 50.0, thresholds=custom)
    assert level == LEVEL_MODERATE
