"""
Model Training and Evaluation Pipeline.
Trains a machine-learning model to predict traffic congestion ~15 minutes into the future
using time-of-day features, road identifiers, current speeds, and historical lags.
Adheres strictly to chronological splitting (no lookahead leakage) and tolerance-based target matching.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import MODELS_DIR, DEMO_DATA_DIR, PROCESSED_DATA_DIR, PUNE_ROAD_POINTS
from src.data_utils import clean_and_prepare_dataset, create_future_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (TrainModel) %(message)s")
logger = logging.getLogger("TrainModel")


def extract_features(df: pd.DataFrame, road_encoder: Optional[LabelEncoder] = None) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Constructs feature matrix from traffic observations:
    - Hour of day (raw & cyclical sin/cos)
    - Day of week & weekend flag
    - Encoded road ID
    - Current speed, free flow speed, congestion ratio, confidence, road closure
    - Speed lag from immediately preceding observation on the same road
    """
    df = df.copy()

    # Time features
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["hour"] = ts.dt.hour + (ts.dt.minute / 60.0)
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical hour encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    # Encode road_id
    if road_encoder is None:
        road_encoder = LabelEncoder()
        # Ensure all known Pune roads are registered
        known_roads = [pt["road_id"] for pt in PUNE_ROAD_POINTS]
        all_roads = list(set(known_roads).union(set(df["road_id"].dropna().unique())))
        road_encoder.fit(all_roads)

    df["road_encoded"] = road_encoder.transform(df["road_id"].astype(str))

    # Basic numeric features
    df["current_speed"] = df["current_speed"].astype(float)
    df["free_flow_speed"] = df["free_flow_speed"].astype(float)
    df["congestion_ratio"] = df["congestion_ratio"].fillna(0.0).astype(float)
    df["confidence"] = df["confidence"].fillna(1.0).astype(float)
    df["road_closure"] = df["road_closure"].astype(int)

    # Calculate lag feature (previous speed on same road)
    df_sorted = df.sort_values(by=["road_id", "timestamp"]).reset_index(drop=True)
    df_sorted["prev_speed"] = df_sorted.groupby("road_id")["current_speed"].shift(1)
    df_sorted["prev_speed"] = df_sorted["prev_speed"].fillna(df_sorted["current_speed"])

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

    return df_sorted[feature_cols], road_encoder


def train_and_evaluate(
    data_path: Optional[Path] = None,
    horizon_minutes: int = 15,
    tolerance_minutes: int = 5,
    test_ratio: float = 0.20
) -> Dict[str, Any]:
    """
    Executes the training and evaluation pipeline:
    1. Loads dataset
    2. Constructs 15-minute future target labels
    3. Splits chronologically
    4. Trains RandomForestClassifier
    5. Evaluates metrics
    6. Saves model & metadata
    """
    # 1. Determine dataset path
    target_data_file = data_path
    if target_data_file is None or not Path(target_data_file).exists():
        # Check processed data first, fallback to demo data
        processed_csv = PROCESSED_DATA_DIR / "pune_traffic_latest_stream.csv"
        demo_csv = DEMO_DATA_DIR / "pune_traffic_historical_demo.csv"

        if processed_csv.exists() and os.path.getsize(processed_csv) > 1024:
            target_data_file = processed_csv
        elif demo_csv.exists():
            target_data_file = demo_csv
        else:
            logger.error("No dataset found in data/processed/ or data/demo/! Run scripts/generate_demo_data.py first.")
            sys.exit(1)

    logger.info("Loading dataset from: %s", target_data_file)
    raw_df = pd.read_csv(target_data_file)
    cleaned_df = clean_and_prepare_dataset(raw_df)

    if len(cleaned_df) < 50:
        logger.error("Insufficient observations (%d rows). Need at least 50 to train model.", len(cleaned_df))
        sys.exit(1)

    is_synthetic = bool((cleaned_df["data_source"] == "synthetic_demo").mean() > 0.5)
    data_label = "Synthetic Demo Dataset" if is_synthetic else "Live TomTom API Dataset"
    logger.info("Dataset loaded: %d valid records. Source: %s", len(cleaned_df), data_label)

    # 2. Construct future target (horizon ~ 15m)
    logger.info("Generating future labels (horizon: %d mins, tolerance: ±%d mins)...",
                horizon_minutes, tolerance_minutes)
    labeled_df = create_future_target(cleaned_df, horizon_minutes=horizon_minutes, tolerance_minutes=tolerance_minutes)

    if labeled_df.empty or len(labeled_df) < 30:
        logger.error(
            "Could not match enough future targets within tolerance window (matched %d rows). "
            "Ensure the dataset has continuous periodic observations.", len(labeled_df)
        )
        sys.exit(1)

    logger.info("Successfully generated %d future-labeled samples.", len(labeled_df))
    logger.info("Target class distribution:\n%s", labeled_df["target_congestion_level"].value_counts())

    # 3. Chronological Train / Test Split
    labeled_df = labeled_df.sort_values(by="timestamp").reset_index(drop=True)
    split_idx = int(len(labeled_df) * (1.0 - test_ratio))

    train_df = labeled_df.iloc[:split_idx].copy()
    test_df = labeled_df.iloc[split_idx:].copy()

    split_time = train_df["timestamp"].iloc[-1]
    logger.info(
        "Chronological Split: %d Train rows (up to %s) | %d Test rows (after %s)",
        len(train_df), split_time, len(test_df), split_time
    )

    # 4. Feature Extraction
    X_train, road_encoder = extract_features(train_df)
    y_train = train_df["target_congestion_level"].values

    X_test, _ = extract_features(test_df, road_encoder=road_encoder)
    y_test = test_df["target_congestion_level"].values

    # 5. Train Baseline Model
    logger.info("Training RandomForestClassifier...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 6. Evaluate Model
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    classes = clf.classes_.tolist()

    logger.info("=" * 60)
    logger.info(" Model Evaluation Results (Horizon: ~15 mins ahead)")
    logger.info(" Evaluation Dataset: %s", data_label)
    logger.info(" Chronological Test Accuracy: %.2f%%", accuracy * 100)
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred, zero_division=0))
    logger.info("Confusion Matrix: %s", conf_matrix)
    logger.info("Classes: %s", classes)
    logger.info("=" * 60)

    # 7. Save Model and Preprocessing Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "congestion_model.joblib"
    encoder_path = MODELS_DIR / "road_encoder.joblib"
    metadata_path = MODELS_DIR / "model_metadata.json"

    joblib.dump(clf, model_path)
    joblib.dump(road_encoder, encoder_path)

    metadata = {
        "model_type": "RandomForestClassifier",
        "horizon_minutes": horizon_minutes,
        "tolerance_minutes": tolerance_minutes,
        "is_synthetic": is_synthetic,
        "data_label": data_label,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "split_timestamp": str(split_time),
        "test_accuracy": round(accuracy, 4),
        "classes": classes,
        "feature_names": list(X_train.columns),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "Model trained on synthetic demonstration data. Real-world accuracy requires "
            "continuous long-term observation collection from live TomTom API."
            if is_synthetic else "Model trained on real historical TomTom API observations."
        )
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model artifacts:")
    logger.info("  Model:    %s", model_path)
    logger.info("  Encoder:  %s", encoder_path)
    logger.info("  Metadata: %s", metadata_path)

    return metadata


class TrafficPredictor:
    """
    Loads saved model and metadata to generate 15-minute future congestion predictions.
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
            return {"error": "Model artifacts not found. Please run train_model.py first."}

        ts = pd.to_datetime(timestamp_utc or datetime.now(timezone.utc), utc=True)
        hour = ts.hour + (ts.minute / 60.0)
        day_of_week = ts.dayofweek
        is_weekend = 1 if day_of_week in [5, 6] else 0
        sin_hour = float(np.sin(2 * np.pi * hour / 24.0))
        cos_hour = float(np.cos(2 * np.pi * hour / 24.0))

        # Safe road encoding (handles unseen roads gracefully)
        try:
            road_encoded = int(self.encoder.transform([road_id])[0])
        except Exception:
            road_encoded = 0

        ratio = round(current_speed / free_flow_speed, 4) if free_flow_speed > 0 else 0.0

        feature_vector = pd.DataFrame([{
            "road_encoded": road_encoded,
            "hour": hour,
            "sin_hour": sin_hour,
            "cos_hour": cos_hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "current_speed": float(current_speed),
            "free_flow_speed": float(free_flow_speed),
            "congestion_ratio": ratio,
            "prev_speed": float(current_speed),
            "confidence": 1.0,
            "road_closure": int(road_closure),
        }])

        pred_class = self.model.predict(feature_vector)[0]
        probabilities = dict(zip(self.model.classes_, [round(float(p), 3) for p in self.model.predict_proba(feature_vector)[0]]))

        return {
            "predicted_congestion_level_15m": pred_class,
            "probabilities": probabilities,
            "road_id": road_id,
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "current_ratio": ratio,
            "is_synthetic_model": self.metadata.get("is_synthetic", True),
            "horizon_minutes": self.metadata.get("horizon_minutes", 15)
        }


def main():
    parser = argparse.ArgumentParser(description="Train historical congestion prediction model.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--horizon", type=int, default=15, help="Prediction horizon in minutes (default: 15)")
    args = parser.parse_args()

    data_file = Path(args.data) if args.data else None
    train_and_evaluate(data_path=data_file, horizon_minutes=args.horizon)


if __name__ == "__main__":
    main()
