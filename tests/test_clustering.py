"""
Unit tests for Traffic K-Means Clustering Module (src/clustering.py).

All synthetic test datasets used below are explicitly marked as synthetic demonstration data
for unit testing purposes.
"""

import math
import pytest
import numpy as np
import pandas as pd

from src.clustering import (
    TrafficKMeansClusterer,
    prepare_clustering_features,
    cluster_traffic_data,
    InsufficientDataError,
    LABEL_HEAVY,
    LABEL_MODERATE,
    LABEL_LIGHT
)


def _generate_synthetic_test_df() -> pd.DataFrame:
    """
    Creates a controlled synthetic test dataset with 3 distinct traffic regimes:
    - 4 Heavy congestion samples (ratio ~0.20 - 0.25)
    - 4 Moderate congestion samples (ratio ~0.50 - 0.55)
    - 4 Light congestion samples (ratio ~0.90 - 1.00)
    """
    records = [
        # Heavy Congestion regime
        {"timestamp": "2026-10-22T01:00:00Z", "road_id": "r1", "current_speed": 10.0, "free_flow_speed": 50.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T01:05:00Z", "road_id": "r2", "current_speed": 11.0, "free_flow_speed": 50.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T01:10:00Z", "road_id": "r3", "current_speed": 12.0, "free_flow_speed": 60.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T01:15:00Z", "road_id": "r4", "current_speed": 8.0, "free_flow_speed": 40.0, "data_source": "synthetic_test"},

        # Moderate Congestion regime
        {"timestamp": "2026-10-22T02:00:00Z", "road_id": "r1", "current_speed": 26.0, "free_flow_speed": 50.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T02:05:00Z", "road_id": "r2", "current_speed": 25.0, "free_flow_speed": 50.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T02:10:00Z", "road_id": "r3", "current_speed": 31.0, "free_flow_speed": 60.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T02:15:00Z", "road_id": "r4", "current_speed": 22.0, "free_flow_speed": 40.0, "data_source": "synthetic_test"},

        # Light Congestion regime
        {"timestamp": "2026-10-22T03:00:00Z", "road_id": "r1", "current_speed": 48.0, "free_flow_speed": 50.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T03:05:00Z", "road_id": "r2", "current_speed": 47.0, "free_flow_speed": 50.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T03:10:00Z", "road_id": "r3", "current_speed": 58.0, "free_flow_speed": 60.0, "data_source": "synthetic_test"},
        {"timestamp": "2026-10-22T03:15:00Z", "road_id": "r4", "current_speed": 39.0, "free_flow_speed": 40.0, "data_source": "synthetic_test"},
    ]
    return pd.DataFrame(records)


def test_prepare_clustering_features_valid():
    """Verify that feature preparation extracts valid numeric features and recomputes ratios."""
    df = _generate_synthetic_test_df()
    features, cleaned = prepare_clustering_features(df)

    assert len(features) == len(df)
    assert set(features.columns) == {"current_speed", "free_flow_speed", "congestion_ratio"}
    assert features["congestion_ratio"].notnull().all()
    assert (features["congestion_ratio"] > 0).all()


def test_prepare_clustering_features_missing_and_invalid():
    """Verify that rows with nulls, non-finite values, and negative speeds are safely excluded."""
    invalid_records = [
        {"timestamp": "2026-10-22T01:00:00Z", "road_id": "r1", "current_speed": None, "free_flow_speed": 50.0},
        {"timestamp": "2026-10-22T01:05:00Z", "road_id": "r2", "current_speed": -10.0, "free_flow_speed": 50.0},
        {"timestamp": "2026-10-22T01:10:00Z", "road_id": "r3", "current_speed": np.inf, "free_flow_speed": 60.0},
        {"timestamp": "2026-10-22T01:15:00Z", "road_id": "r4", "current_speed": 25.0, "free_flow_speed": np.nan},
        # One valid row
        {"timestamp": "2026-10-22T01:20:00Z", "road_id": "r5", "current_speed": 30.0, "free_flow_speed": 50.0},
    ]
    df_invalid = pd.DataFrame(invalid_records)
    features, cleaned = prepare_clustering_features(df_invalid)

    assert len(features) == 1
    assert cleaned.iloc[0]["road_id"] == "r5"
    assert features.iloc[0]["congestion_ratio"] == 0.60


def test_zero_free_flow_speed_handling():
    """Verify that free_flow_speed <= 0 prevents division by zero and is safely filtered."""
    zero_free_records = [
        {"timestamp": "2026-10-22T01:00:00Z", "road_id": "r1", "current_speed": 20.0, "free_flow_speed": 0.0},
        {"timestamp": "2026-10-22T01:05:00Z", "road_id": "r2", "current_speed": 20.0, "free_flow_speed": -5.0},
        {"timestamp": "2026-10-22T01:10:00Z", "road_id": "r3", "current_speed": 25.0, "free_flow_speed": 50.0},
    ]
    df = pd.DataFrame(zero_free_records)
    features, cleaned = prepare_clustering_features(df)

    assert len(features) == 1
    assert cleaned.iloc[0]["road_id"] == "r3"
    assert features.iloc[0]["free_flow_speed"] == 50.0


def test_insufficient_observations_handling():
    """Verify that fitting with fewer samples than clusters raises InsufficientDataError."""
    df_small = pd.DataFrame([
        {"timestamp": "2026-10-22T01:00:00Z", "road_id": "r1", "current_speed": 20.0, "free_flow_speed": 50.0},
        {"timestamp": "2026-10-22T01:05:00Z", "road_id": "r2", "current_speed": 25.0, "free_flow_speed": 50.0},
    ])
    clusterer = TrafficKMeansClusterer(n_clusters=3)

    with pytest.raises(InsufficientDataError):
        clusterer.fit(df_small)

    # Verify high-level helper returns graceful error message instead of raising unhandled exception
    annotated, stats, model, err = cluster_traffic_data(df_small, n_clusters=3)
    assert annotated.empty
    assert err is not None
    assert "Insufficient observations" in err


def test_cluster_assignment_and_labels_interpretation():
    """
    Verify that K-Means correctly assigns 3 clusters and dynamically interprets
    clusters based on learned characteristics (congestion ratio and speed).
    """
    df = _generate_synthetic_test_df()
    clusterer = TrafficKMeansClusterer(n_clusters=3, random_state=42)
    annotated = clusterer.fit_predict(df)

    assert len(annotated) == len(df)
    assert "cluster_id" in annotated.columns
    assert "cluster_label" in annotated.columns

    # Check distinct cluster labels
    assigned_labels = set(annotated["cluster_label"].unique())
    assert assigned_labels == {LABEL_HEAVY, LABEL_MODERATE, LABEL_LIGHT}

    # Verify cluster statistics
    stats = clusterer.get_cluster_statistics()
    assert len(stats) == 3

    # Check that Heavy Congestion has the lowest mean congestion ratio
    heavy_stat = stats[stats["cluster_label"] == LABEL_HEAVY].iloc[0]
    moderate_stat = stats[stats["cluster_label"] == LABEL_MODERATE].iloc[0]
    light_stat = stats[stats["cluster_label"] == LABEL_LIGHT].iloc[0]

    assert heavy_stat["mean_congestion_ratio"] < moderate_stat["mean_congestion_ratio"]
    assert moderate_stat["mean_congestion_ratio"] < light_stat["mean_congestion_ratio"]


def test_reproducibility_with_fixed_seed():
    """Verify that multiple runs with the same random_state yield identical cluster assignments."""
    df = _generate_synthetic_test_df()

    c1 = TrafficKMeansClusterer(n_clusters=3, random_state=42)
    res1 = c1.fit_predict(df)

    c2 = TrafficKMeansClusterer(n_clusters=3, random_state=42)
    res2 = c2.fit_predict(df)

    assert (res1["cluster_id"].values == res2["cluster_id"].values).all()
    assert (res1["cluster_label"].values == res2["cluster_label"].values).all()


def test_cluster_statistics_and_centroids():
    """Verify that cluster statistics and centroids return well-formed DataFrames."""
    df = _generate_synthetic_test_df()
    clusterer = TrafficKMeansClusterer(n_clusters=3, random_state=42).fit(df)

    stats = clusterer.get_cluster_statistics()
    centroids = clusterer.get_centroids()

    assert not stats.empty
    assert "count" in stats.columns
    assert "mean_current_speed" in stats.columns
    assert "mean_congestion_ratio" in stats.columns

    assert not centroids.empty
    assert "cluster_id" in centroids.columns
    assert "cluster_label" in centroids.columns
    assert "current_speed" in centroids.columns
    assert "congestion_ratio" in centroids.columns


def test_model_persistence_and_loading(tmp_path):
    """Verify that a fitted clusterer can be saved to joblib and loaded back with identical predictions."""
    df = _generate_synthetic_test_df()
    clusterer = TrafficKMeansClusterer(n_clusters=3, random_state=42).fit(df)

    save_path = tmp_path / "test_kmeans.joblib"
    clusterer.save(save_path)
    assert save_path.exists()

    loaded = TrafficKMeansClusterer.load(save_path)
    assert loaded.is_fitted
    assert loaded.cluster_mapping == clusterer.cluster_mapping

    pred_original = clusterer.predict(df)
    pred_loaded = loaded.predict(df)
    assert (pred_original["cluster_id"].values == pred_loaded["cluster_id"].values).all()
    assert (pred_original["cluster_label"].values == pred_loaded["cluster_label"].values).all()


def test_cluster_traffic_data_empty_or_none():
    """Verify that high-level helper returns graceful empty outputs on empty or None input."""
    # Test None
    ann, stats, model, err = cluster_traffic_data(None)
    assert ann.empty
    assert stats.empty
    assert model is None
    assert "No traffic observations" in err

    # Test empty DataFrame
    ann, stats, model, err = cluster_traffic_data(pd.DataFrame())
    assert ann.empty
    assert stats.empty
    assert model is None
    assert "No traffic observations" in err


def test_dashboard_fallback_insufficient_or_corrupt_data():
    """
    Verify dashboard helper gracefully handles corrupt inputs, non-numeric values,
    and missing columns without unhandled exceptions.
    """
    # Corrupt data with non-numeric speed strings
    corrupt_df = pd.DataFrame([
        {"road_id": "r1", "current_speed": "corrupt_speed", "free_flow_speed": "invalid"},
        {"road_id": "r2", "current_speed": "N/A", "free_flow_speed": 40.0},
    ])
    ann, stats, model, err = cluster_traffic_data(corrupt_df, n_clusters=3)
    assert ann.empty
    assert stats.empty
    assert model is None
    assert err is not None
    assert "Insufficient observations" in err or "Clustering error" in err

    # Missing critical speed columns entirely
    missing_cols_df = pd.DataFrame([
        {"road_id": "r1", "latitude": 18.5, "longitude": 73.8},
        {"road_id": "r2", "latitude": 18.6, "longitude": 73.9},
    ])
    ann, stats, model, err = cluster_traffic_data(missing_cols_df, n_clusters=3)
    assert ann.empty
    assert stats.empty
    assert err is not None
