"""
Model Training and Evaluation Pipeline.
Trains a machine-learning model to predict traffic congestion ~15 minutes into the future
using time-of-day features, road identifiers, current speeds, and historical lags.
Adheres strictly to chronological splitting (no lookahead leakage) and tolerance-based target matching.

Delegates core implementation to `src.congestion_prediction` for modular architecture.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.congestion_prediction import (
    extract_features,
    train_and_evaluate_congestion_model as train_and_evaluate,
    CongestionPredictor,
    TrafficPredictor
)


def main():
    parser = argparse.ArgumentParser(description="Train historical congestion prediction model.")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--horizon", type=int, default=15, help="Prediction horizon in minutes (default: 15)")
    parser.add_argument("--tolerance", type=int, default=5, help="Tolerance window in minutes (default: ±5)")
    parser.add_argument("--test-ratio", type=float, default=0.20, help="Test ratio (default: 0.20)")
    args = parser.parse_args()

    data_file = Path(args.data) if args.data else None
    train_and_evaluate(
        data_path=data_file,
        horizon_minutes=args.horizon,
        tolerance_minutes=args.tolerance,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
