"""
Unit tests for Isolation Forest Anomaly Detection Module.
(src/anomaly_detection.py)

Uses synthetic test fixtures explicitly labeled as synthetic demo data.
"""

import pytest
import numpy as np
import pandas as pd

from src.anomaly_detection import (
    prepare_anomaly_features,
    TrafficAnomalyDetector,
    detect_traffic_anomalies
)


def _generate_synthetic_anomaly_fixture() -> pd.DataFrame:
    """
    Creates a synthetic fixture with predominantly normal observations
    and a few injected extreme anomalies (e.g. crawling 3 km/h on a 65 km/h highway).
    """
    records = []
    # 30 normal observations across roads
    for i in range(30):
        records.append({
            "timestamp": f"2026-10-22T08:{i:02d}:00Z",
            "road_id": f"road_{i % 3}",
            "current_speed": 42.0 + (i % 6),
            "free_flow_speed": 50.0,
            "data_source": "synthetic_test"
        })

    # Injected severe anomalies (severe speed drop / bottleneck)
    records.append({
        "timestamp": "2026-10-22T09:00:00Z",
        "road_id": "road_0",
        "current_speed": 4.0,  # Extreme drop from 50 km/h baseline
        "free_flow_speed": 50.0,
        "data_source": "synthetic_test"
    })
    records.append({
        "timestamp": "2026-10-22T09:05:00Z",
        "road_id": "road_1",
        "current_speed": 3.0,  # Extreme drop from 60 km/h baseline
        "free_flow_speed": 60.0,
        "data_source": "synthetic_test"
    })

    return pd.DataFrame(records)


def test_prepare_anomaly_features_valid():
    """Verify feature engineering extracts speed_deficit and congestion_ratio."""
    df = _generate_synthetic_anomaly_fixture()
    features, cleaned = prepare_anomaly_features(df)

    assert len(features) == len(df)
    assert set(features.columns) == {"current_speed", "free_flow_speed", "congestion_ratio", "speed_deficit"}
    assert (features["speed_deficit"] >= 0).all()
    assert (features["congestion_ratio"] > 0).all()


def test_prepare_anomaly_features_invalid_and_zero():
    """Verify rows with zero free-flow speed or missing speeds are safely screened out."""
    df_invalid = pd.DataFrame([
        {"timestamp": "2026-10-22T01:00:00Z", "road_id": "r1", "current_speed": 20.0, "free_flow_speed": 0.0},
        {"timestamp": "2026-10-22T01:05:00Z", "road_id": "r2", "current_speed": None, "free_flow_speed": 50.0},
        {"timestamp": "2026-10-22T01:10:00Z", "road_id": "r3", "current_speed": 30.0, "free_flow_speed": 50.0},
    ])
    features, cleaned = prepare_anomaly_features(df_invalid)

    assert len(features) == 1
    assert cleaned.iloc[0]["road_id"] == "r3"


def test_isolation_forest_fit_predict_detection():
    """Verify Isolation Forest flags injected severe anomalies."""
    df = _generate_synthetic_anomaly_fixture()
    detector = TrafficAnomalyDetector(contamination=0.08, random_state=42)
    scored_df = detector.fit_predict(df)

    assert len(scored_df) == len(df)
    assert "is_anomaly" in scored_df.columns
    assert "anomaly_label" in scored_df.columns
    assert "anomaly_score" in scored_df.columns

    # The extreme drop to 3 km/h on a 60 km/h road should be identified as anomalous
    extreme_row = scored_df[scored_df["current_speed"] == 3.0].iloc[0]
    assert extreme_row["is_anomaly"] == True
    assert extreme_row["anomaly_label"] == "Anomalous"
    assert extreme_row["anomaly_score"] > 0.50


def test_isolation_forest_persistence(tmp_path):
    """Verify model saving and loading preserves prediction behavior."""
    df = _generate_synthetic_anomaly_fixture()
    detector = TrafficAnomalyDetector(contamination=0.08, random_state=42).fit(df)

    model_file = tmp_path / "iso_forest_test.joblib"
    meta_file = tmp_path / "iso_meta_test.json"
    detector.save(model_file, meta_file)

    assert model_file.exists()
    assert meta_file.exists()

    loaded = TrafficAnomalyDetector.load(model_file)
    assert loaded.is_fitted
    assert loaded.contamination == detector.contamination

    pred_orig = detector.predict(df)
    pred_load = loaded.predict(df)

    assert (pred_orig["is_anomaly"].values == pred_load["is_anomaly"].values).all()
    assert np.allclose(pred_orig["anomaly_score"].values, pred_load["anomaly_score"].values)


def test_detect_traffic_anomalies_empty_and_fallback():
    """Verify high-level helper returns graceful status on empty or invalid DataFrame."""
    ann, stats, det, err = detect_traffic_anomalies(pd.DataFrame())
    assert ann.empty
    assert stats == {}
    assert err is not None
    assert "No traffic observations" in err
