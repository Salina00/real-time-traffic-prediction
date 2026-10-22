"""
Data utility functions:
- Validation of traffic records
- CSV / Parquet dataset cleaning and preparation
- Chronological sorting and duplicate elimination
- Strict tolerance-based future target generation (15-min congestion prediction)
- Appending raw observations
"""

import os
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

from config.settings import RAW_DATA_DIR, CONGESTION_THRESHOLDS
from src.congestion import calculate_congestion_ratio, classify_congestion

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = [
    "timestamp",
    "road_id",
    "latitude",
    "longitude",
    "current_speed",
    "free_flow_speed"
]


def validate_traffic_record(record: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validates an incoming traffic observation dictionary.
    Returns (is_valid, error_message).
    """
    if not isinstance(record, dict):
        return False, "Record is not a dictionary"

    # 1. Check required fields
    for field in REQUIRED_FIELDS:
        if field not in record or record[field] is None:
            return False, f"Missing required field: '{field}'"

    # 2. Check timestamp format
    ts_val = record["timestamp"]
    if not isinstance(ts_val, (str, datetime)):
        return False, f"Invalid timestamp type: {type(ts_val)}"
    if isinstance(ts_val, str):
        try:
            # Parse ISO 8601
            datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except ValueError:
            return False, f"Timestamp is not valid ISO 8601: '{ts_val}'"

    # 3. Check coordinates
    try:
        lat = float(record["latitude"])
        lon = float(record["longitude"])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            return False, f"Coordinates out of bounds: lat={lat}, lon={lon}"
    except (ValueError, TypeError):
        return False, "Non-numeric latitude or longitude"

    # 4. Check speeds
    try:
        cur_spd = float(record["current_speed"])
        free_spd = float(record["free_flow_speed"])
        is_closed = bool(record.get("road_closure", False))

        if cur_spd < 0:
            return False, f"Current speed cannot be negative: {cur_spd}"
        if free_spd <= 0 and not is_closed:
            return False, f"Free-flow speed must be positive: {free_spd}"
    except (ValueError, TypeError):
        return False, "Non-numeric current_speed or free_flow_speed"

    return True, None


def clean_and_prepare_dataset(
    data: Union[pd.DataFrame, str, Path, List[Union[str, Path]]],
    output_filepath: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Loads, cleans, validates, and standardizes traffic records:
    - Parses timestamps to datetime in UTC
    - Removes duplicate observations (same road_id & timestamp)
    - Re-computes congestion_ratio and congestion_level to guarantee consistency
    - Sorts chronologically
    - Saves to output_filepath if provided
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {p}")
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    elif isinstance(data, list):
        dfs = []
        for p in data:
            path_obj = Path(p)
            if path_obj.exists():
                if path_obj.suffix == ".parquet":
                    dfs.append(pd.read_parquet(path_obj))
                else:
                    dfs.append(pd.read_csv(path_obj))
        if not dfs:
            raise ValueError("No valid files found in list to load")
        df = pd.concat(dfs, ignore_index=True)
    else:
        raise TypeError(f"Unsupported data type for cleaning: {type(data)}")

    if df.empty:
        return df

    # Drop rows missing essential identifiers or speeds
    df = df.dropna(subset=["timestamp", "road_id", "current_speed", "free_flow_speed"]).copy()

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Ensure numeric types
    for col in ["current_speed", "free_flow_speed", "latitude", "longitude", "confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "road_closure" in df.columns:
        df["road_closure"] = df["road_closure"].astype(bool)
    else:
        df["road_closure"] = False

    # Filter out invalid speeds
    df = df[df["current_speed"] >= 0]
    df = df[(df["free_flow_speed"] > 0) | (df["road_closure"] == True)]

    # Remove duplicates
    df = df.drop_duplicates(subset=["road_id", "timestamp"])

    # Sort chronologically
    df = df.sort_values(by=["timestamp", "road_id"]).reset_index(drop=True)

    # Recompute congestion ratio and congestion level
    levels = []
    ratios = []
    for _, row in df.iterrows():
        ratio, level = classify_congestion(
            current_speed=row["current_speed"],
            free_flow_speed=row["free_flow_speed"],
            road_closure=row.get("road_closure", False),
            thresholds=CONGESTION_THRESHOLDS
        )
        ratios.append(ratio)
        levels.append(level)

    df["congestion_ratio"] = ratios
    df["congestion_level"] = levels

    # Ensure data_source is present
    if "data_source" not in df.columns:
        df["data_source"] = "unknown"

    if output_filepath:
        out_p = Path(output_filepath)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix == ".parquet":
            df.to_parquet(out_p, index=False)
        else:
            df.to_csv(out_p, index=False)
        logger.info("Saved cleaned dataset (%d rows) to %s", len(df), out_p)

    return df


def create_future_target(
    df: pd.DataFrame,
    horizon_minutes: int = 15,
    tolerance_minutes: int = 5
) -> pd.DataFrame:
    """
    Constructs future prediction targets for each road point without lookahead leakage:
    For an observation at time t on road R, searches for a future observation
    on the same road R at time t_future such that:
        (horizon - tolerance) <= (t_future - t) <= (horizon + tolerance)

    Parameters:
        df: Cleaned chronological dataframe of observations
        horizon_minutes: Target lookahead horizon (default 15 mins)
        tolerance_minutes: Allowed delta window (default ±5 mins, i.e. 10 to 20 mins)

    Returns:
        DataFrame containing input features at time t, plus:
        - target_timestamp
        - target_speed
        - target_congestion_ratio
        - target_congestion_level (the primary ML target label)
    """
    if df.empty:
        return pd.DataFrame()

    df_sorted = df.sort_values(by=["road_id", "timestamp"]).copy()
    min_delta = pd.Timedelta(minutes=horizon_minutes - tolerance_minutes)
    max_delta = pd.Timedelta(minutes=horizon_minutes + tolerance_minutes)

    matched_rows = []

    # Process each road independently
    for road_id, group in df_sorted.groupby("road_id"):
        group = group.reset_index(drop=True)
        timestamps = group["timestamp"].values
        n = len(group)

        for i in range(n):
            current_t = group.loc[i, "timestamp"]
            current_row = group.loc[i].to_dict()

            target_delta = pd.Timedelta(minutes=horizon_minutes)
            best_match_idx = None
            best_diff = None

            for j in range(i + 1, n):
                delta = group.loc[j, "timestamp"] - current_t
                if delta < min_delta:
                    continue
                if delta > max_delta:
                    # Beyond tolerance window, no need to look further ahead
                    break

                diff = abs((delta - target_delta).total_seconds())
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_match_idx = j

            if best_match_idx is not None:
                future_row = group.loc[best_match_idx]
                current_row["target_timestamp"] = future_row["timestamp"]
                current_row["target_speed"] = future_row["current_speed"]
                current_row["target_congestion_ratio"] = future_row["congestion_ratio"]
                current_row["target_congestion_level"] = future_row["congestion_level"]
                matched_rows.append(current_row)

    if not matched_rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(matched_rows)
    return result_df.sort_values(by="timestamp").reset_index(drop=True)


def save_raw_observation_to_csv(record: Dict[str, Any], filepath: Optional[Union[str, Path]] = None):
    """
    Thread-safe / incremental appending of raw observations to CSV.
    Useful for capturing live TomTom data for future model retraining.
    """
    target = Path(filepath or (RAW_DATA_DIR / "traffic_raw_events.csv"))
    target.parent.mkdir(parents=True, exist_ok=True)

    file_exists = target.exists()
    keys = list(record.keys())

    with open(target, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
