"""
Supervised Traffic Congestion Prediction Module using Random Forest.

Trains a machine-learning model to forecast traffic congestion ~15 minutes into the future
using time-of-day features, road identifiers, current speeds, and historical lags.

Methodological Highlights:
- Strict chronological train/test splitting (prevents lookahead leakage).
- Evaluates against a naive Majority-Class Baseline (DummyClassifier).
- Generates precision, recall, F1-score, and confusion matrix.
- Balanced class weights to handle severe congestion imbalance.
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Suppress joblib physical cores warning on Windows
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

from config.settings import (
    MODELS_DIR, DEMO_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR,
    PUNE_ROAD_POINTS, CONGESTION_MODEL_PATH, ROAD_ENCODER_PATH, MODEL_METADATA_PATH
)
from src.data_utils import clean_and_prepare_dataset, create_future_target

logger = logging.getLogger(__name__)


def extract_features(
    df: pd.DataFrame,
    road_encoder: Optional[LabelEncoder] = None
) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Constructs feature matrix from traffic observations:
    - Hour of day (raw & cyclical sin/cos)
    - Day of week & weekend flag
    - Properly encoded road ID (via LabelEncoder)
    - Current speed, free flow speed, congestion ratio, confidence, road closure
    - Speed lag from preceding observation on the same road (historical lag)
    """
    df = df.copy()

    # 1. Temporal features
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour + (ts.dt.minute / 60.0)
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical hour encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # 2. Encode road_id properly
    if road_encoder is None:
        road_encoder = LabelEncoder()
        known_roads = [pt["road_id"] for pt in PUNE_ROAD_POINTS]
        all_roads = sorted(list(set(known_roads).union(set(df["road_id"].dropna().astype(str).unique()))))
        road_encoder.fit(all_roads)

    df["road_encoded"] = road_encoder.transform(df["road_id"].astype(str))

    # 3. Numeric features with division-by-zero & missing value guards
    df["current_speed"] = pd.to_numeric(df["current_speed"], errors="coerce").fillna(0.0).astype(float)
    df["free_flow_speed"] = pd.to_numeric(df["free_flow_speed"], errors="coerce").fillna(50.0).clip(lower=1.0).astype(float)
    
    # Calculate congestion ratio safely
    if "congestion_ratio" not in df.columns or df["congestion_ratio"].isnull().any():
        df["congestion_ratio"] = (df["current_speed"] / df["free_flow_speed"]).round(4)
    else:
        df["congestion_ratio"] = pd.to_numeric(df["congestion_ratio"], errors="coerce").fillna(
            (df["current_speed"] / df["free_flow_speed"])
        ).round(4).astype(float)

    df["confidence"] = pd.to_numeric(df.get("confidence", 1.0), errors="coerce").fillna(1.0).astype(float)
    df["road_closure"] = df.get("road_closure", False).astype(int)

    # 4. Calculate lag feature (previous observed speed on same road) without scrambling row index
    sorted_idx = df.sort_values(by=["road_id", "timestamp"]).index
    prev_speed_series = df.loc[sorted_idx].groupby("road_id")["current_speed"].shift(1)
    df["prev_speed"] = prev_speed_series.reindex(df.index).fillna(df["current_speed"]).astype(float)

    feature_cols = [
        "road_encoded",
        "hour",
        "sin_hour",
        "cos_hour",
        "day_of_week",
        "is_weekend",
        "current_speed",
        "free_flow_speed",
        "congestion_ratio",
        "prev_speed",
        "confidence",
        "road_closure",
    ]

    return df[feature_cols].copy(), road_encoder


def train_and_evaluate_congestion_model(
    data_path: Optional[Union[str, Path]] = None,
    horizon_minutes: int = 15,
    tolerance_minutes: int = 5,
    test_ratio: float = 0.20,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes the complete supervised training and evaluation pipeline:
    1. Loads and cleans dataset.
    2. Constructs future targets within (horizon ± tolerance) window.
    3. Splits strictly chronologically (prevents lookahead data leakage).
    4. Evaluates a DummyClassifier majority-class baseline.
    5. Trains RandomForestClassifier with class_weight='balanced'.
    6. Computes comprehensive metrics (accuracy, precision, recall, F1, confusion matrix).
    7. Persists trained model, label encoder, and evaluation metadata.
    """
    target_data_file = Path(data_path) if data_path else None
    if target_data_file is None or not target_data_file.exists():
        processed_csv = PROCESSED_DATA_DIR / "pune_traffic_latest_stream.csv"
        demo_csv = DEMO_DATA_DIR / "pune_traffic_historical_demo.csv"
        raw_csv = RAW_DATA_DIR / "traffic_raw_events.csv"

        if demo_csv.exists() and demo_csv.stat().st_size > 1024:
            target_data_file = demo_csv
        elif processed_csv.exists() and processed_csv.stat().st_size > 1024:
            target_data_file = processed_csv
        elif raw_csv.exists() and raw_csv.stat().st_size > 100:
            target_data_file = raw_csv
        else:
            raise FileNotFoundError("No traffic dataset found! Run scripts/generate_demo_data.py first.")

    logger.info("Loading dataset from: %s", target_data_file)
    raw_df = pd.read_parquet(target_data_file) if target_data_file.suffix == ".parquet" else pd.read_csv(target_data_file)
    cleaned_df = clean_and_prepare_dataset(raw_df)

    if len(cleaned_df) < 50:
        raise ValueError(f"Insufficient observations ({len(cleaned_df)} rows). Need at least 50 to train model.")

    is_synthetic = bool((cleaned_df["data_source"] == "synthetic_demo").mean() > 0.5)
    data_label = "Synthetic Demo Dataset" if is_synthetic else "Live TomTom API Dataset"

    # 2. Construct future target labels (tolerance matching without lookahead)
    logger.info("Constructing future targets (horizon: %d mins, tolerance: ±%d mins)...",
                horizon_minutes, tolerance_minutes)
    labeled_df = create_future_target(
        cleaned_df,
        horizon_minutes=horizon_minutes,
        tolerance_minutes=tolerance_minutes
    )

    if labeled_df.empty or len(labeled_df) < 30:
        raise ValueError(
            f"Could not match enough future targets within tolerance window (matched {len(labeled_df)} rows). "
            "Ensure the dataset has periodic chronological observations."
        )

    # 3. Chronological Train / Test Split
    labeled_df = labeled_df.sort_values(by="timestamp").reset_index(drop=True)
    split_idx = int(len(labeled_df) * (1.0 - test_ratio))

    train_df = labeled_df.iloc[:split_idx].copy()
    test_df = labeled_df.iloc[split_idx:].copy()
    split_time = str(train_df["timestamp"].iloc[-1])

    # 4. Extract Features
    X_train, road_encoder = extract_features(train_df)
    y_train = train_df["target_congestion_level"].values

    X_test, _ = extract_features(test_df, road_encoder=road_encoder)
    y_test = test_df["target_congestion_level"].values

    # 5. Baseline Model (Majority-Class DummyClassifier)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    baseline_accuracy = float(accuracy_score(y_test, y_pred_dummy))

    # 6. Train Random Forest Classifier
    logger.info("Training RandomForestClassifier (n_estimators=100, class_weight='balanced')...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=random_state,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 7. Evaluate Model
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    classes = clf.classes_.tolist()

    # 8. Persist Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CONGESTION_MODEL_PATH)
    joblib.dump(road_encoder, ROAD_ENCODER_PATH)

    metadata = {
        "model_type": "RandomForestClassifier",
        "horizon_minutes": horizon_minutes,
        "tolerance_minutes": tolerance_minutes,
        "is_synthetic": is_synthetic,
        "data_label": data_label,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "split_timestamp": split_time,
        "test_accuracy": round(accuracy, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "classes": classes,
        "feature_names": list(X_train.columns),
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Model trained on synthetic demonstration data. Real-world accuracy requires "
            "continuous long-term observation collection from live TomTom API."
            if is_synthetic else "Model trained on real historical TomTom API observations."
        )
    }

    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info(" Random Forest Supervised Congestion Forecaster")
    logger.info(" Training Samples:      %d | Test Samples: %d", len(train_df), len(test_df))
    logger.info(" Majority Baseline Acc: %.2f%%", baseline_accuracy * 100)
    logger.info(" Model Test Accuracy:   %.2f%%", accuracy * 100)
    logger.info(" Macro F1-Score:        %.4f", macro_f1)
    logger.info(" Classes:               %s", classes)
    logger.info("=" * 60)

    return metadata


class CongestionPredictor:
    """
    Inference client for the trained 15-minute Random Forest forecaster.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        target_dir = models_dir or MODELS_DIR
        self.model_path = target_dir / "congestion_model.joblib"
        self.encoder_path = target_dir / "road_encoder.joblib"
        self.metadata_path = target_dir / "model_metadata.json"

        self.model = None
        self.encoder = None
        self.metadata = None

        if self.is_ready():
            self.load()

    def is_ready(self) -> bool:
        return self.model_path.exists() and self.encoder_path.exists() and self.metadata_path.exists()

    def load(self):
        self.model = joblib.load(self.model_path)
        self.encoder = joblib.load(self.encoder_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def predict(
        self,
        road_id: str,
        current_speed: float,
        free_flow_speed: float,
        timestamp_utc: Optional[str] = None,
        road_closure: bool = False
    ) -> Dict[str, Any]:
        """
        Predicts future congestion level (~15 mins ahead) for a given road observation.
        """
        if not self.is_ready():
            return {"error": "Model artifacts not found. Please train model first."}

        ts = pd.to_datetime(timestamp_utc or datetime.now(timezone.utc), utc=True)
        hour = ts.hour + (ts.minute / 60.0)
        day_of_week = ts.dayofweek
        is_weekend = 1 if day_of_week in [5, 6] else 0
        sin_hour = float(np.sin(2 * np.pi * hour / 24.0))
        cos_hour = float(np.cos(2 * np.pi * hour / 24.0))

        # Safe road encoding (handles unseen roads gracefully)
        try:
            road_encoded = int(self.encoder.transform([str(road_id)])[0])
        except Exception:
            road_encoded = 0

        free_safe = max(float(free_flow_speed), 1.0)
        ratio = round(float(current_speed) / free_safe, 4)

        feature_vector = pd.DataFrame([{
            "road_encoded": road_encoded,
            "hour": hour,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "current_speed": float(current_speed),
            "free_flow_speed": free_safe,
            "congestion_ratio": ratio,
            "prev_speed": float(current_speed),
            "confidence": 1.0,
            "road_closure": int(road_closure),
        }])

        pred_class = self.model.predict(feature_vector)[0]
        prob_values = self.model.predict_proba(feature_vector)[0]
        probabilities = dict(zip(self.model.classes_, [round(float(p), 3) for p in prob_values]))

        return {
            "predicted_congestion_level_15m": pred_class,
            "probabilities": probabilities,
            "road_id": road_id,
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "current_ratio": ratio,
            "is_synthetic_model": self.metadata.get("is_synthetic", True),
            "horizon_minutes": self.metadata.get("horizon_minutes", 15),
            "forecast_timestamp_utc": (ts + pd.Timedelta(minutes=self.metadata.get("horizon_minutes", 15))).isoformat()
        }


# Maintain backward compatibility alias
TrafficPredictor = CongestionPredictor
