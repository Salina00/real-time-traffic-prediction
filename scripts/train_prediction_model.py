"""
Workflow Script: Train Supervised 15-Minute Traffic Congestion Prediction Model.

Trains a RandomForestClassifier using chronological splitting, tolerance-based future target matching,
and evaluates against a majority-class baseline.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.congestion_prediction import train_and_evaluate_congestion_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (TrainPredictionModel) %(message)s")
logger = logging.getLogger("TrainPredictionModel")


def main():
    parser = argparse.ArgumentParser(description="Train 15-Minute Supervised Congestion Forecaster")
    parser.add_argument("--data", type=str, default=None, help="Path to input dataset (CSV/Parquet)")
    parser.add_argument("--horizon", type=int, default=15, help="Forecast horizon in minutes (default: 15)")
    parser.add_argument("--tolerance", type=int, default=5, help="Matching tolerance window in minutes (default: ±5)")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test split ratio (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed (default: 42)")
    args = parser.parse_args()

    data_file = Path(args.data) if args.data else None
    train_and_evaluate_congestion_model(
        data_path=data_file,
        horizon_minutes=args.horizon,
        tolerance_minutes=args.tolerance,
        test_ratio=args.test_ratio,
        random_state=args.seed
    )


if __name__ == "__main__":
    main()
