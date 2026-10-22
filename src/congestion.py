"""
Congestion classification module.
Provides reusable functions to calculate congestion ratio and classify
congestion levels based on configurable thresholds.
"""

import math
from typing import Optional, Tuple, Dict, Any
from config.settings import CONGESTION_THRESHOLDS

# Defined congestion levels
LEVEL_LOW = "low"
LEVEL_MODERATE = "moderate"
LEVEL_HIGH = "high"
LEVEL_CLOSED = "closed"
LEVEL_UNKNOWN = "unknown"

VALID_LEVELS = [LEVEL_LOW, LEVEL_MODERATE, LEVEL_HIGH, LEVEL_CLOSED, LEVEL_UNKNOWN]


def calculate_congestion_ratio(
    current_speed: Optional[float],
    free_flow_speed: Optional[float]
) -> Optional[float]:
    """
    Calculates the speed congestion ratio:
        congestion_ratio = current_speed / free_flow_speed

    Safely handles:
    - None or NaN values
    - free_flow_speed <= 0 (avoids division by zero or negative ratio anomalies)
    - negative current_speed

    Returns:
        float rounded to 4 decimals, or None if inputs are invalid.
    """
    if current_speed is None or free_flow_speed is None:
        return None

    try:
        cur = float(current_speed)
        free = float(free_flow_speed)
    except (ValueError, TypeError):
        return None

    if math.isnan(cur) or math.isnan(free):
        return None

    if free <= 0 or cur < 0:
        return None

    ratio = cur / free
    # Round to 4 decimal places for precision without float drift
    return round(ratio, 4)


def classify_congestion(
    current_speed: Optional[float],
    free_flow_speed: Optional[float],
    road_closure: bool = False,
    thresholds: Optional[Dict[str, float]] = None
) -> Tuple[Optional[float], str]:
    """
    Classifies traffic congestion into project-defined categories:
    - road_closure is True: 'closed' (overrides any speed ratio)
    - ratio >= low_min (default 0.70): 'low'
    - moderate_min <= ratio < low_min (default 0.40 - 0.70): 'moderate'
    - ratio < moderate_min (default < 0.40): 'high'
    - missing or invalid measurements: 'unknown'

    Parameters:
        current_speed: Observed speed in KMPH
        free_flow_speed: Uncongested baseline speed in KMPH
        road_closure: Boolean indicating physical road closure
        thresholds: Optional custom dict with 'low_min' and 'moderate_min'

    Returns:
        (congestion_ratio, congestion_level)
    """
    # 1. Road closure overrides speed calculations
    if bool(road_closure):
        # We can still compute ratio if speeds are available, but level is closed
        ratio = calculate_congestion_ratio(current_speed, free_flow_speed)
        return (ratio, LEVEL_CLOSED)

    # 2. Calculate ratio
    ratio = calculate_congestion_ratio(current_speed, free_flow_speed)
    if ratio is None:
        return (None, LEVEL_UNKNOWN)

    # 3. Apply configurable thresholds
    active_thresholds = thresholds or CONGESTION_THRESHOLDS
    low_min = active_thresholds.get("low_min", 0.70)
    mod_min = active_thresholds.get("moderate_min", 0.40)

    if ratio >= low_min:
        return (ratio, LEVEL_LOW)
    elif ratio >= mod_min:
        return (ratio, LEVEL_MODERATE)
    else:
        return (ratio, LEVEL_HIGH)


def enrich_record_with_congestion(record: Dict[str, Any], thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Convenience function that takes a raw traffic observation dict
    and computes 'congestion_ratio' and 'congestion_level' in-place.
    """
    cur_speed = record.get("current_speed")
    free_speed = record.get("free_flow_speed")
    closure = record.get("road_closure", False)

    ratio, level = classify_congestion(cur_speed, free_speed, road_closure=closure, thresholds=thresholds)
    record["congestion_ratio"] = ratio
    record["congestion_level"] = level
    return record
