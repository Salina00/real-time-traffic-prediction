"""
Workflow Script: Train and Persist Isolation Forest Anomaly Detection Model.

Trains an IsolationForest model on representative traffic observations
(demo dataset or accumulated streaming observations) and persists artifacts
to `models/isolation_forest_model.joblib` and `models/anomaly_metadata.json`.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from config.settings import (
    PROCESSED_DATA_DIR, DEMO_DATA_DIR, RAW_DATA_DIR,
    ISOLATION_FOREST_CONTAMINATION, ISOLATION_FOREST_RANDOM_STATE
)
from src.anomaly_detection import TrafficAnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (TrainAnomalyModel) %(message)s")
logger = logging.getLogger("TrainAnomalyModel")


def main():
    parser = argparse.ArgumentParser(description="Train Isolation Forest Anomaly Detection Model")
    parser.add_argument("--data", type=str, default=None, help="Path to input dataset (CSV/Parquet)")
    parser.add_argument("--contamination", type=float, default=ISOLATION_FOREST_CONTAMINATION, help="Expected outlier proportion (default: 0.05)")
    parser.add_argument("--seed", type=int, default=ISOLATION_FOREST_RANDOM_STATE, help="Random state seed (default: 42)")
    args = parser.parse_args()

    # Determine input dataset
    input_file = None
    if args.data and Path(args.data).exists():
        input_file = Path(args.data)
    else:
        stream_csv = PROCESSED_DATA_DIR / "pune_traffic_latest_stream.csv"
        demo_csv = DEMO_DATA_DIR / "pune_traffic_historical_demo.csv"
        raw_csv = RAW_DATA_DIR / "traffic_raw_events.csv"

        if demo_csv.exists() and demo_csv.stat().st_size > 1024:
            input_file = demo_csv
        elif stream_csv.exists() and stream_csv.stat().st_size > 1024:
            input_file = stream_csv
        elif raw_csv.exists() and raw_csv.stat().st_size > 100:
            input_file = raw_csv

    if not input_file or not input_file.exists():
        logger.error("No dataset found for training Isolation Forest! Run scripts/generate_demo_data.py first.")
        sys.exit(1)

    logger.info("Loading baseline training data from: %s", input_file)
    df = pd.read_parquet(input_file) if input_file.suffix == ".parquet" else pd.read_csv(input_file)
    logger.info("Loaded %d raw observations.", len(df))

    detector = TrafficAnomalyDetector(
        contamination=args.contamination,
        random_state=args.seed
    )
    detector.fit(df)
    detector.save()

    # Quick validation run
    scored_df = detector.predict(df)
    anom_count = int(scored_df["is_anomaly"].sum())
    total_count = len(scored_df)
    anom_pct = (anom_count / total_count) * 100

    logger.info("=" * 60)
    logger.info(" Isolation Forest Training Complete")
    logger.info(" Training Samples:      %d", total_count)
    logger.info(" Contamination Factor:  %.3f (%.1f%%)", args.contamination, args.contamination * 100)
    logger.info(" Flagged Anomalies:     %d (%.2f%%)", anom_count, anom_pct)
    logger.info(" Severe Anomalies:      %d", int((scored_df["anomaly_severity"] == "Severe Anomaly").sum()))
    logger.info(" Data Source:           %s", detector.data_source_label)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
