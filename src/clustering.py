"""
Traffic K-Means Clustering Module (Unsupervised Data Mining).

Provides unsupervised pattern discovery for urban traffic conditions by grouping
multidimensional traffic observations into distinct congestion clusters:
- Heavy Congestion
- Moderate Congestion
- Light Congestion

Unlike supervised forecasting (which predicts traffic 15 minutes ahead), K-Means
identifies latent structural groupings in current or historical observations
without relying on ground-truth target labels.

Features Used:
- current_speed (km/h): Observed traffic speed
- free_flow_speed (km/h): Uncongested baseline road speed
- congestion_ratio: current_speed / free_flow_speed (normalized traffic throughput)
"""

import os
import sys
import logging
import warnings
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Prevent joblib physical cores warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config.settings import (
    KMEANS_DEFAULT_CLUSTERS,
    KMEANS_RANDOM_STATE,
    KMEANS_MIN_OBSERVATIONS,
    KMEANS_FEATURE_COLS,
    KMEANS_MODEL_PATH
)
from src.congestion import calculate_congestion_ratio

logger = logging.getLogger(__name__)

# Standard Semantic Labels for Congestion Clusters
LABEL_HEAVY = "Heavy Congestion"
LABEL_MODERATE = "Moderate Congestion"
LABEL_LIGHT = "Light Congestion"


class InsufficientDataError(ValueError):
    """Raised when there are not enough valid traffic observations to fit K-Means."""
    pass


def prepare_clustering_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans, validates, and extracts numerical features for traffic clustering.

    Steps:
    1. Validates DataFrame structure and presence of essential columns.
    2. Drops rows with missing or non-finite values (NaN, inf, -inf).
    3. Prevents division-by-zero by safely recomputing or validating congestion_ratio.
    4. Filters out physically invalid speeds (current_speed < 0 or free_flow_speed <= 0).
    5. Excludes categorical fields (e.g., road_name, timestamp) from feature matrix.
    6. Optionally removes duplicate observations (same road_id and timestamp).

    Parameters:
        df: Input DataFrame containing traffic observations.
        feature_cols: List of numerical columns to use. Defaults to KMEANS_FEATURE_COLS.
        drop_duplicates: Whether to remove duplicate observations.

    Returns:
        Tuple of (clean_feature_df, cleaned_full_df):
        - clean_feature_df: DataFrame containing only the selected numerical features.
        - cleaned_full_df: Full DataFrame subset corresponding to the valid rows.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Empty or invalid DataFrame passed to prepare_clustering_features.")
        return pd.DataFrame(), pd.DataFrame()

    cols = feature_cols or list(KMEANS_FEATURE_COLS)
    working_df = df.copy()

    # 1. Remove duplicate observations if timestamp & road_id are available
    if drop_duplicates and "road_id" in working_df.columns and "timestamp" in working_df.columns:
        working_df = working_df.drop_duplicates(subset=["road_id", "timestamp"]).copy()

    # 2. Ensure speed columns exist and are numeric
    for speed_col in ["current_speed", "free_flow_speed"]:
        if speed_col not in working_df.columns:
            logger.error("Required column '%s' missing from dataset.", speed_col)
            return pd.DataFrame(), pd.DataFrame()
        working_df[speed_col] = pd.to_numeric(working_df[speed_col], errors="coerce")

    # 3. Handle or recalculate congestion_ratio safely to prevent division-by-zero
    if "congestion_ratio" not in working_df.columns or working_df["congestion_ratio"].isnull().any():
        computed_ratios = []
        for _, row in working_df.iterrows():
            ratio = calculate_congestion_ratio(row["current_speed"], row["free_flow_speed"])
            computed_ratios.append(ratio)
        working_df["congestion_ratio"] = computed_ratios
    else:
        working_df["congestion_ratio"] = pd.to_numeric(working_df["congestion_ratio"], errors="coerce")

    # 4. Filter out invalid rows:
    # - current_speed must be non-negative (>= 0)
    # - free_flow_speed must be strictly positive (> 0) to avoid division by zero
    # - congestion_ratio must be valid and finite
    valid_mask = (
        working_df["current_speed"].notnull() &
        (working_df["current_speed"] >= 0) &
        working_df["free_flow_speed"].notnull() &
        (working_df["free_flow_speed"] > 0) &
        working_df["congestion_ratio"].notnull() &
        np.isfinite(working_df["current_speed"]) &
        np.isfinite(working_df["free_flow_speed"]) &
        np.isfinite(working_df["congestion_ratio"])
    )
    cleaned_full_df = working_df[valid_mask].copy()

    if cleaned_full_df.empty:
        logger.warning("No valid observations remained after cleaning.")
        return pd.DataFrame(), pd.DataFrame()

    # 5. Extract only the selected numerical features
    clean_feature_df = cleaned_full_df[cols].astype(float).copy()

    return clean_feature_df, cleaned_full_df


class TrafficKMeansClusterer:
    """
    Unsupervised Traffic Pattern Discovery using K-Means Clustering.

    Workflow:
    1. Preprocesses and scales input traffic features using StandardScaler.
    2. Fits scikit-learn KMeans with reproducible random_state and n_init.
    3. Examines learned cluster centroids (mean congestion_ratio and mean speed).
    4. Dynamically maps arbitrary cluster IDs (0, 1, 2) to semantic traffic states:
       - Lowest congestion ratio (slowest relative throughput) -> 'Heavy Congestion'
       - Intermediate congestion ratio                         -> 'Moderate Congestion'
       - Highest congestion ratio (near free-flow speed)       -> 'Light Congestion'
    5. Computes comprehensive cluster statistics for dashboard visualization.
    """

    def __init__(
        self,
        n_clusters: int = KMEANS_DEFAULT_CLUSTERS,
        random_state: int = KMEANS_RANDOM_STATE,
        n_init: int = 10,
        min_observations: int = KMEANS_MIN_OBSERVATIONS,
        feature_cols: Optional[List[str]] = None
    ):
        """
        Initializes the K-Means Traffic Clusterer.

        Parameters:
            n_clusters: Number of clusters (default: 3).
            random_state: Seed for reproducible centroid initialization.
            n_init: Number of centroid seeding runs.
            min_observations: Minimum sample count required for reliable fitting.
            feature_cols: List of numerical columns used for clustering.
        """
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be at least 1, got {n_clusters}")

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.min_observations = min_observations
        self.feature_cols = feature_cols or list(KMEANS_FEATURE_COLS)

        self.scaler: Optional[StandardScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.cluster_mapping: Dict[int, str] = {}
        self.cluster_stats: List[Dict[str, Any]] = []
        self.is_fitted: bool = False
        self.training_sample_count: int = 0
        self.data_source_label: str = "unknown"

    def fit(self, df: pd.DataFrame) -> "TrafficKMeansClusterer":
        """
        Fits StandardScaler and KMeans on valid traffic observations.

        Raises:
            InsufficientDataError: If valid observations are fewer than n_clusters.
        """
        features_df, cleaned_df = prepare_clustering_features(df, feature_cols=self.feature_cols)
        n_samples = len(features_df)

        if n_samples < self.n_clusters:
            raise InsufficientDataError(
                f"Insufficient observations to fit {self.n_clusters} clusters. "
                f"Found {n_samples} valid rows, but need at least {self.n_clusters}."
            )

        if n_samples < self.min_observations:
            warnings.warn(
                f"Fitting K-Means on a small dataset ({n_samples} samples < threshold {self.min_observations}). "
                "Cluster boundaries may be sensitive to outliers.",
                UserWarning
            )
            logger.warning("Fitting K-Means on small sample size: %d observations.", n_samples)

        # Record dataset source (TomTom vs Synthetic Demo)
        if "data_source" in cleaned_df.columns:
            sources = cleaned_df["data_source"].dropna().unique().tolist()
            if len(sources) == 1 and sources[0] == "tomtom":
                self.data_source_label = "Live TomTom Observations"
            elif all("demo" in str(s).lower() or "synthetic" in str(s).lower() for s in sources):
                self.data_source_label = "Synthetic Demo Dataset"
            else:
                self.data_source_label = f"Mixed ({', '.join(sources)})"
        else:
            self.data_source_label = "Historical Observations"

        # 1. Feature Scaling using StandardScaler
        self.scaler = StandardScaler()
        # Handle potential zero-variance constant columns safely
        X_scaled = self.scaler.fit_transform(features_df)

        # 2. Fit KMeans
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init
        )
        raw_cluster_ids = self.kmeans.fit_predict(X_scaled)

        # 3. Interpret clusters dynamically by examining unscaled feature means
        self.cluster_mapping, self.cluster_stats = self._interpret_clusters(
            features_df=features_df,
            raw_cluster_ids=raw_cluster_ids
        )

        self.is_fitted = True
        self.training_sample_count = n_samples
        logger.info(
            "TrafficKMeansClusterer fitted successfully on %d observations (K=%d). Mapping: %s",
            n_samples, self.n_clusters, self.cluster_mapping
        )
        return self

    def _interpret_clusters(
        self,
        features_df: pd.DataFrame,
        raw_cluster_ids: np.ndarray
    ) -> Tuple[Dict[int, str], List[Dict[str, Any]]]:
        """
        Examines learned cluster characteristics and assigns meaningful labels.

        Logic:
        - Calculates mean `congestion_ratio` and mean `current_speed` for each cluster.
        - Clusters with lowest congestion ratios correspond to severe traffic bottlenecks.
        - Clusters with highest congestion ratios correspond to smooth, free-flowing roads.
        - Ranks clusters so that labels reflect traffic physics rather than random numeric IDs.
        """
        temp_df = features_df.copy()
        temp_df["cluster_id"] = raw_cluster_ids

        cluster_summaries = []
        for c_id in range(self.n_clusters):
            c_mask = temp_df["cluster_id"] == c_id
            c_rows = temp_df[c_mask]

            count = len(c_rows)
            if count > 0:
                mean_ratio = float(c_rows["congestion_ratio"].mean())
                mean_speed = float(c_rows["current_speed"].mean())
                mean_free_flow = float(c_rows["free_flow_speed"].mean())
            else:
                mean_ratio = 0.0
                mean_speed = 0.0
                mean_free_flow = 0.0

            cluster_summaries.append({
                "cluster_id": c_id,
                "count": count,
                "percentage": round((count / len(temp_df)) * 100, 1) if len(temp_df) > 0 else 0.0,
                "mean_congestion_ratio": round(mean_ratio, 4),
                "mean_current_speed": round(mean_speed, 2),
                "mean_free_flow_speed": round(mean_free_flow, 2),
            })

        # Rank clusters by congestion ratio ascending (lowest ratio = most congested).
        # In case of tie in ratio, break ties with mean_current_speed ascending.
        sorted_by_congestion = sorted(
            cluster_summaries,
            key=lambda x: (x["mean_congestion_ratio"], x["mean_current_speed"])
        )

        mapping: Dict[int, str] = {}
        if self.n_clusters == 3:
            # Standard 3-class assignment
            mapping[sorted_by_congestion[0]["cluster_id"]] = LABEL_HEAVY
            mapping[sorted_by_congestion[1]["cluster_id"]] = LABEL_MODERATE
            mapping[sorted_by_congestion[2]["cluster_id"]] = LABEL_LIGHT
        elif self.n_clusters == 2:
            mapping[sorted_by_congestion[0]["cluster_id"]] = LABEL_HEAVY
            mapping[sorted_by_congestion[1]["cluster_id"]] = LABEL_LIGHT
        elif self.n_clusters == 1:
            mapping[sorted_by_congestion[0]["cluster_id"]] = LABEL_MODERATE
        else:
            # Generic K > 3 ranking: lowest = heavy, highest = light, intermediate = moderate tiers
            total = len(sorted_by_congestion)
            for rank, item in enumerate(sorted_by_congestion):
                c_id = item["cluster_id"]
                if rank == 0:
                    mapping[c_id] = LABEL_HEAVY
                elif rank == total - 1:
                    mapping[c_id] = LABEL_LIGHT
                else:
                    mapping[c_id] = f"{LABEL_MODERATE} (Tier {rank})"

        # Attach assigned semantic labels into the summary list
        for summary in cluster_summaries:
            summary["cluster_label"] = mapping.get(summary["cluster_id"], "Unknown")

        # Sort summary output by cluster_id for consistent presentation
        cluster_summaries = sorted(cluster_summaries, key=lambda x: x["cluster_id"])
        return mapping, cluster_summaries

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns cluster IDs and semantic labels to observations using fitted model.

        Parameters:
            df: Input DataFrame with traffic records.

        Returns:
            DataFrame containing original columns plus 'cluster_id' and 'cluster_label'.
        """
        if not self.is_fitted or self.kmeans is None or self.scaler is None:
            raise RuntimeError("Clusterer has not been fitted yet. Call fit() first.")

        features_df, cleaned_df = prepare_clustering_features(df, feature_cols=self.feature_cols)
        if features_df.empty:
            return pd.DataFrame()

        # Scale using the existing scaler
        X_scaled = self.scaler.transform(features_df)
        raw_cluster_ids = self.kmeans.predict(X_scaled)

        result_df = cleaned_df.copy()
        result_df["cluster_id"] = raw_cluster_ids
        result_df["cluster_label"] = [
            self.cluster_mapping.get(cid, "Unknown") for cid in raw_cluster_ids
        ]

        return result_df

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to fit on dataset and return annotated DataFrame."""
        return self.fit(df).predict(df)

    def get_cluster_statistics(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing statistics for each learned cluster:
        cluster_id, cluster_label, observation count, percentage, and feature means.
        """
        if not self.is_fitted:
            return pd.DataFrame()
        return pd.DataFrame(self.cluster_stats)

    def get_centroids(self) -> pd.DataFrame:
        """
        Returns cluster centroids converted back into original physical units
        (km/h and ratio) via the inverse scaler.
        """
        if not self.is_fitted or self.kmeans is None or self.scaler is None:
            return pd.DataFrame()

        # Inverse transform scaled centroids back to original feature scale
        unscaled_centroids = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(unscaled_centroids, columns=self.feature_cols)
        centroids_df["cluster_id"] = list(range(len(centroids_df)))
        centroids_df["cluster_label"] = [
            self.cluster_mapping.get(cid, "Unknown") for cid in centroids_df["cluster_id"]
        ]

        # Round for readability
        for col in self.feature_cols:
            if col == "congestion_ratio":
                centroids_df[col] = centroids_df[col].round(4)
            else:
                centroids_df[col] = centroids_df[col].round(2)

        return centroids_df

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Persists the fitted clusterer, scaler, and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted clusterer.")

        out_path = Path(filepath or KMEANS_MODEL_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "n_clusters": self.n_clusters,
            "random_state": self.random_state,
            "n_init": self.n_init,
            "feature_cols": self.feature_cols,
            "scaler": self.scaler,
            "kmeans": self.kmeans,
            "cluster_mapping": self.cluster_mapping,
            "cluster_stats": self.cluster_stats,
            "training_sample_count": self.training_sample_count,
            "data_source_label": self.data_source_label
        }
        joblib.dump(artifact, out_path)
        logger.info("Saved K-Means clustering artifacts to %s", out_path)
        return out_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "TrafficKMeansClusterer":
        """Loads a previously persisted clusterer from disk."""
        in_path = Path(filepath or KMEANS_MODEL_PATH)
        if not in_path.exists():
            raise FileNotFoundError(f"K-Means model artifact not found at {in_path}")

        artifact = joblib.load(in_path)
        instance = cls(
            n_clusters=artifact.get("n_clusters", KMEANS_DEFAULT_CLUSTERS),
            random_state=artifact.get("random_state", KMEANS_RANDOM_STATE),
            n_init=artifact.get("n_init", 10),
            feature_cols=artifact.get("feature_cols", KMEANS_FEATURE_COLS)
        )
        instance.scaler = artifact.get("scaler")
        instance.kmeans = artifact.get("kmeans")
        instance.cluster_mapping = artifact.get("cluster_mapping", {})
        instance.cluster_stats = artifact.get("cluster_stats", [])
        instance.training_sample_count = artifact.get("training_sample_count", 0)
        instance.data_source_label = artifact.get("data_source_label", "Loaded Artifact")
        instance.is_fitted = True
        return instance


def cluster_traffic_data(
    df: pd.DataFrame,
    n_clusters: int = KMEANS_DEFAULT_CLUSTERS,
    random_state: int = KMEANS_RANDOM_STATE,
    min_observations: int = KMEANS_MIN_OBSERVATIONS,
    persist_model: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[TrafficKMeansClusterer], Optional[str]]:
    """
    High-level, defensive helper function for dashboard and script integration.

    Executes complete clustering workflow safely:
    - Preprocesses and validates input observations.
    - Fits K-Means with dynamic cluster interpretation.
    - Returns annotated DataFrame, cluster statistics, clusterer instance, and status message.
    - Does not crash if data is insufficient or invalid.

    Returns:
        (annotated_df, cluster_stats_df, clusterer_instance, error_message)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), pd.DataFrame(), None, "No traffic observations provided."

    clusterer = TrafficKMeansClusterer(
        n_clusters=n_clusters,
        random_state=random_state,
        min_observations=min_observations
    )

    try:
        annotated_df = clusterer.fit_predict(df)
        stats_df = clusterer.get_cluster_statistics()

        if persist_model:
            try:
                clusterer.save()
            except Exception as e:
                logger.warning("Could not persist K-Means model: %s", e)

        return annotated_df, stats_df, clusterer, None

    except InsufficientDataError as ide:
        logger.warning("Clustering skipped: %s", ide)
        return pd.DataFrame(), pd.DataFrame(), None, str(ide)
    except Exception as e:
        logger.error("Unexpected error during traffic clustering: %s", e, exc_info=True)
        return pd.DataFrame(), pd.DataFrame(), None, f"Clustering error: {str(e)}"


if __name__ == "__main__":
    import argparse
    from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, DEMO_DATA_DIR

    parser = argparse.ArgumentParser(description="Run K-Means Traffic Clustering on historical or streaming data.")
    parser.add_argument("--k", type=int, default=KMEANS_DEFAULT_CLUSTERS, help="Number of clusters (default: 3)")
    parser.add_argument("--data", type=str, default=None, help="Path to input CSV or Parquet dataset")
    parser.add_argument("--save", action="store_true", default=True, help="Save fitted model to models/kmeans_traffic_model.joblib")
    parser.add_argument("--export-csv", action="store_true", default=True, help="Save clustered records to data/processed/clustered_traffic_latest.csv")
    args = parser.parse_args()

    # Determine data source
    input_path = None
    if args.data and Path(args.data).exists():
        input_path = Path(args.data)
    else:
        # Check standard pipeline locations
        stream_csv = PROCESSED_DATA_DIR / "pune_traffic_latest_stream.csv"
        raw_csv = RAW_DATA_DIR / "traffic_raw_events.csv"
        demo_csv = DEMO_DATA_DIR / "pune_traffic_historical_demo.csv"

        if stream_csv.exists() and stream_csv.stat().st_size > 100:
            input_path = stream_csv
        elif raw_csv.exists() and raw_csv.stat().st_size > 100:
            input_path = raw_csv
        elif demo_csv.exists():
            input_path = demo_csv

    if not input_path or not input_path.exists():
        print("[ERROR] No traffic dataset found! Run producer.py or scripts/generate_demo_data.py first.")
        exit(1)

    print(f"--> Loading dataset from: {input_path}")
    df_in = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    print(f"--> Loaded {len(df_in)} raw rows.")

    annotated, stats, model, err = cluster_traffic_data(
        df=df_in,
        n_clusters=args.k,
        persist_model=args.save
    )

    if err:
        print(f"[FAILED] {err}")
        exit(1)

    print("\n" + "=" * 65)
    print(f" K-Means Traffic Clustering Results (K={args.k})")
    print(f" Data Source:      {model.data_source_label}")
    print(f" Samples Clustered: {len(annotated)}")
    print("=" * 65)

    print("\nCluster Statistics:")
    print(stats.to_string(index=False))

    print("\nLearned Centroids (Original Physical Units):")
    centroids = model.get_centroids()
    print(centroids.to_string(index=False))

    if args.export_csv:
        out_csv = PROCESSED_DATA_DIR / "clustered_traffic_latest.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        annotated.to_csv(out_csv, index=False)
        print(f"\n--> Exported clustered dataset to: {out_csv}")

