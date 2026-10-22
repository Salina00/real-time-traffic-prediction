"""
Traffic Anomaly Detection Module using Isolation Forest.

Identifies traffic observations that deviate significantly from learned normal
patterns (e.g., unexpected sudden speed drops, severe bottlenecks relative to
road baseline capacity, or unusual sensor readings).

IMPORTANT METHODOLOGICAL NOTE:
Anomaly detection flags statistical deviations and unusual traffic flow patterns;
it does NOT determine the physical cause (e.g., whether an anomaly was caused by
an accident, bad weather, construction, or VIP convoy movement).
"""

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Suppress joblib physical cores warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from config.settings import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_RANDOM_STATE,
    ISOLATION_FOREST_MODEL_PATH,
    ISOLATION_FOREST_METADATA_PATH,
    ISOLATION_FOREST_FEATURE_COLS
)
from src.congestion import calculate_congestion_ratio

logger = logging.getLogger(__name__)


def prepare_anomaly_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    drop_duplicates: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans, validates, and engineers numerical features for anomaly detection:
    - current_speed (km/h)
    - free_flow_speed (km/h)
    - congestion_ratio (current_speed / free_flow_speed)
    - speed_deficit (free_flow_speed - current_speed, positive when traffic slows)

    Defensive protections:
    - Filters out missing (NaN/None) and non-finite (inf/-inf) values.
    - Prevents division-by-zero by enforcing free_flow_speed > 0 and current_speed >= 0.
    - Excludes categorical fields (road names, identifiers) from numerical distance metrics.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.warning("Empty or invalid DataFrame passed to prepare_anomaly_features.")
        return pd.DataFrame(), pd.DataFrame()

    working_df = df.copy()

    # 1. Optionally deduplicate
    if drop_duplicates and "road_id" in working_df.columns and "timestamp" in working_df.columns:
        working_df = working_df.drop_duplicates(subset=["road_id", "timestamp"]).copy()

    # 2. Ensure speed columns exist and are numeric
    for col in ["current_speed", "free_flow_speed"]:
        if col not in working_df.columns:
            logger.error("Required column '%s' missing from dataset.", col)
            return pd.DataFrame(), pd.DataFrame()
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    # 3. Compute or validate congestion_ratio safely
    if "congestion_ratio" not in working_df.columns or working_df["congestion_ratio"].isnull().any():
        computed_ratios = []
        for _, row in working_df.iterrows():
            ratio = calculate_congestion_ratio(row["current_speed"], row["free_flow_speed"])
            computed_ratios.append(ratio)
        working_df["congestion_ratio"] = computed_ratios
    else:
        working_df["congestion_ratio"] = pd.to_numeric(working_df["congestion_ratio"], errors="coerce")

    # 4. Filter invalid speeds & non-finite numbers
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
    cleaned_df = working_df[valid_mask].copy()

    if cleaned_df.empty:
        logger.warning("No valid observations remained after cleaning for anomaly detection.")
        return pd.DataFrame(), pd.DataFrame()

    # 5. Derive speed deficit (baseline free flow speed minus observed speed)
    cleaned_df["speed_deficit"] = (
        cleaned_df["free_flow_speed"] - cleaned_df["current_speed"]
    ).clip(lower=0.0).round(2)

    cols = feature_cols or list(ISOLATION_FOREST_FEATURE_COLS)
    feature_df = cleaned_df[cols].astype(float).copy()

    return feature_df, cleaned_df


class TrafficAnomalyDetector:
    """
    Isolation Forest Anomaly Detector for Urban Traffic Patterns.

    Workflow:
    1. Preprocesses speed and congestion metrics.
    2. Scales features using StandardScaler for uniform tree-partitioning balance.
    3. Fits IsolationForest with reproducible random_state and configurable contamination.
    4. Computes anomaly indicators (-1: anomaly, 1: normal) and continuous anomaly scores.
    5. Classifies severity into 'Normal', 'Moderate Anomaly', and 'Severe Anomaly'.
    """

    def __init__(
        self,
        contamination: float = ISOLATION_FOREST_CONTAMINATION,
        random_state: int = ISOLATION_FOREST_RANDOM_STATE,
        n_estimators: int = 100,
        feature_cols: Optional[List[str]] = None
    ):
        if not (0.0 < contamination <= 0.5):
            raise ValueError(f"Contamination must be between 0.0 and 0.5, got {contamination}")

        self.contamination = contamination
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.feature_cols = feature_cols or list(ISOLATION_FOREST_FEATURE_COLS)

        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False
        self.training_sample_count: int = 0
        self.data_source_label: str = "unknown"
        self.score_mean: float = 0.0
        self.score_std: float = 1.0

    def fit(self, df: pd.DataFrame) -> "TrafficAnomalyDetector":
        """Fits the Isolation Forest on representative traffic baseline data."""
        features_df, cleaned_df = prepare_anomaly_features(df, feature_cols=self.feature_cols)

        if len(features_df) < 10:
            raise ValueError(
                f"Insufficient data to train Isolation Forest. Need at least 10 valid rows, got {len(features_df)}."
            )

        # Record dataset source
        if "data_source" in cleaned_df.columns:
            sources = cleaned_df["data_source"].dropna().unique().tolist()
            if len(sources) == 1 and sources[0] == "tomtom":
                self.data_source_label = "Live TomTom Observations"
            elif all("demo" in str(s).lower() or "synthetic" in str(s).lower() for s in sources):
                self.data_source_label = "Synthetic Demo Dataset"
            else:
                self.data_source_label = f"Mixed ({', '.join(sources)})"
        else:
            self.data_source_label = "Historical Baseline"

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features_df)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators
        )
        self.model.fit(X_scaled)

        # Compute baseline decision score parameters for normalization
        decision_scores = self.model.decision_function(X_scaled)
        self.score_mean = float(np.mean(decision_scores))
        self.score_std = float(np.std(decision_scores)) or 1.0

        self.is_fitted = True
        self.training_sample_count = len(features_df)
        logger.info(
            "TrafficAnomalyDetector fitted on %d samples (contamination=%.3f, source=%s).",
            self.training_sample_count, self.contamination, self.data_source_label
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs anomaly inference on input traffic records.

        Returns DataFrame containing original columns plus:
        - raw_anomaly_score: float (decision_function, lower/negative indicates anomaly)
        - anomaly_score: float (normalized 0.0 to 1.0, higher = more anomalous)
        - is_anomaly: bool (True if flagged anomalous)
        - anomaly_label: str ('Anomalous' vs 'Normal')
        - anomaly_severity: str ('Severe Anomaly', 'Moderate Anomaly', 'Normal')
        """
        if not self.is_fitted or self.model is None or self.scaler is None:
            raise RuntimeError("Anomaly detector has not been fitted. Call fit() or load() first.")

        features_df, cleaned_df = prepare_anomaly_features(df, feature_cols=self.feature_cols)
        if features_df.empty:
            return pd.DataFrame()

        X_scaled = self.scaler.transform(features_df)
        raw_preds = self.model.predict(X_scaled)  # -1 for anomaly, 1 for inlier
        raw_scores = self.model.decision_function(X_scaled)

        result_df = cleaned_df.copy()
        result_df["raw_anomaly_score"] = raw_scores.round(4)

        # Normalized anomaly score: 0.0 (very normal) to 1.0 (extreme anomaly)
        # Using sigmoid-like transformation of standardized decision scores
        z_scores = -(raw_scores - self.score_mean) / self.score_std
        normalized_scores = 1.0 / (1.0 + np.exp(-z_scores))
        result_df["anomaly_score"] = np.round(normalized_scores, 4)

        # Flag and labels
        is_anom = raw_preds == -1
        result_df["is_anomaly"] = is_anom
        result_df["anomaly_label"] = np.where(is_anom, "Anomalous", "Normal")

        # Severity categorization
        severities = []
        for anom, score in zip(is_anom, normalized_scores):
            if not anom:
                severities.append("Normal")
            elif score >= 0.75:
                severities.append("Severe Anomaly")
            else:
                severities.append("Moderate Anomaly")
        result_df["anomaly_severity"] = severities

        return result_df

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method to fit on dataset and return scored observations."""
        return self.fit(df).predict(df)

    def save(self, filepath: Optional[Union[str, Path]] = None, meta_path: Optional[Union[str, Path]] = None) -> Path:
        """Persists trained model artifacts and metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted anomaly detector.")

        out_path = Path(filepath or ISOLATION_FOREST_MODEL_PATH)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "feature_cols": self.feature_cols,
            "scaler": self.scaler,
            "model": self.model,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "training_sample_count": self.training_sample_count,
            "data_source_label": self.data_source_label
        }
        joblib.dump(artifact, out_path)

        # Save human-readable metadata JSON
        json_path = Path(meta_path or ISOLATION_FOREST_METADATA_PATH)
        metadata = {
            "model_type": "IsolationForest",
            "contamination": self.contamination,
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "feature_cols": self.feature_cols,
            "training_sample_count": self.training_sample_count,
            "data_source_label": self.data_source_label,
            "explanation": (
                "Isolation Forest isolates anomalies by randomly partitioning feature trees. "
                "Observations requiring fewer splittings are flagged as anomalies. "
                "Detected anomalies represent unusual traffic conditions, not their physical causes."
            )
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved Isolation Forest artifact to %s and metadata to %s", out_path, json_path)
        return out_path

    @classmethod
    def load(cls, filepath: Optional[Union[str, Path]] = None) -> "TrafficAnomalyDetector":
        """Loads a previously fitted anomaly detector from disk."""
        in_path = Path(filepath or ISOLATION_FOREST_MODEL_PATH)
        if not in_path.exists():
            raise FileNotFoundError(f"Isolation Forest model artifact not found at {in_path}")

        artifact = joblib.load(in_path)
        instance = cls(
            contamination=artifact.get("contamination", ISOLATION_FOREST_CONTAMINATION),
            random_state=artifact.get("random_state", ISOLATION_FOREST_RANDOM_STATE),
            n_estimators=artifact.get("n_estimators", 100),
            feature_cols=artifact.get("feature_cols", ISOLATION_FOREST_FEATURE_COLS)
        )
        instance.scaler = artifact.get("scaler")
        instance.model = artifact.get("model")
        instance.score_mean = artifact.get("score_mean", 0.0)
        instance.score_std = artifact.get("score_std", 1.0)
        instance.training_sample_count = artifact.get("training_sample_count", 0)
        instance.data_source_label = artifact.get("data_source_label", "Loaded Model")
        instance.is_fitted = True
        return instance


def detect_traffic_anomalies(
    df: pd.DataFrame,
    contamination: float = ISOLATION_FOREST_CONTAMINATION,
    random_state: int = ISOLATION_FOREST_RANDOM_STATE,
    use_saved_model: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[TrafficAnomalyDetector], Optional[str]]:
    """
    High-level, defensive helper function for dashboard and script integration.

    Executes anomaly detection safely:
    - Attempts to use saved model artifact if present; otherwise fits on provided data.
    - Returns annotated DataFrame, anomaly statistics summary, detector, and error message.
    - Does not crash if data is empty or invalid.

    Returns:
        (annotated_df, summary_stats_dict, detector_instance, error_message)
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(), {}, None, "No traffic observations provided for anomaly detection."

    detector = None

    # 1. Try loading saved model first if requested
    if use_saved_model and ISOLATION_FOREST_MODEL_PATH.exists():
        try:
            detector = TrafficAnomalyDetector.load(ISOLATION_FOREST_MODEL_PATH)
        except Exception as e:
            logger.warning("Could not load saved Isolation Forest model (%s), fitting dynamic model.", e)
            detector = None

    # 2. If no saved model or load failed, fit on available data
    if detector is None:
        try:
            detector = TrafficAnomalyDetector(contamination=contamination, random_state=random_state)
            detector.fit(df)
        except Exception as e:
            logger.error("Failed to fit Isolation Forest: %s", e)
            return pd.DataFrame(), {}, None, f"Anomaly detection fitting error: {str(e)}"

    # 3. Perform inference
    try:
        annotated_df = detector.predict(df)
        if annotated_df.empty:
            return pd.DataFrame(), {}, detector, "No valid observations remained after cleaning."

        total_obs = len(annotated_df)
        anomaly_count = int(annotated_df["is_anomaly"].sum())
        normal_count = total_obs - anomaly_count
        anomaly_pct = round((anomaly_count / total_obs) * 100, 2) if total_obs > 0 else 0.0

        stats = {
            "total_observations": total_obs,
            "anomaly_count": anomaly_count,
            "normal_count": normal_count,
            "anomaly_percentage": anomaly_pct,
            "contamination_parameter": detector.contamination,
            "data_source_label": detector.data_source_label,
            "severe_anomaly_count": int((annotated_df["anomaly_severity"] == "Severe Anomaly").sum()),
            "moderate_anomaly_count": int((annotated_df["anomaly_severity"] == "Moderate Anomaly").sum()),
        }

        return annotated_df, stats, detector, None

    except Exception as e:
        logger.error("Error during anomaly prediction: %s", e)
        return pd.DataFrame(), {}, detector, f"Anomaly inference error: {str(e)}"
