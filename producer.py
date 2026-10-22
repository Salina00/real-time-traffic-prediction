"""
Traffic Data Producer.
Fetches real-time traffic flow data from TomTom API (or generates realistic demo events)
for Pune road points and publishes each observation as a JSON event to Kafka.
Also supports optional raw CSV persistence for historical data collection.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    TOMTOM_API_KEY, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC,
    COLLECTION_INTERVAL_SECONDS, DATA_MODE, PUNE_ROAD_POINTS,
    RAW_DATA_DIR, LOGS_DIR
)
from src.tomtom_client import TomTomTrafficClient, mask_key
from src.kafka_producer import TrafficKafkaProducer
from src.data_utils import save_raw_observation_to_csv

# Setup logger with console & file handlers
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Producer) %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "producer.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("TrafficProducer")


def run_collection_cycle(
    client: TomTomTrafficClient,
    kafka_producer: TrafficKafkaProducer,
    mode: str,
    save_csv: bool = True
) -> int:
    """
    Executes a single observation cycle across all Pune road points.
    Returns the count of successfully processed events.
    """
    logger.info("Starting traffic data collection cycle (Mode: %s)...", mode.upper())
    observations = []

    if mode == "tomtom":
        if not client.is_api_key_available():
            logger.error(
                "TomTom mode selected, but TOMTOM_API_KEY is not set in .env! "
                "Falling back to synthetic demo events for this cycle."
            )
            observations = client.generate_demo_snapshot()
        else:
            try:
                observations = client.fetch_pune_snapshot()
                if not observations:
                    logger.warning("No live observations returned from TomTom API. Falling back to demo data.")
                    observations = client.generate_demo_snapshot()
            except Exception as e:
                logger.error("Error fetching live TomTom snapshot: %s. Falling back to demo.", e)
                observations = client.generate_demo_snapshot()
    else:
        # Explicit demo mode
        observations = client.generate_demo_snapshot()

    logger.info("Obtained %d observations for cycle.", len(observations))

    success_count = 0
    for obs in observations:
        # 1. Optionally save to raw CSV for auditing & training dataset accumulation
        if save_csv:
            try:
                save_raw_observation_to_csv(obs, filepath=RAW_DATA_DIR / "traffic_raw_events.csv")
            except Exception as csv_err:
                logger.warning("Failed to append raw CSV: %s", csv_err)

        # 2. Publish JSON event to Kafka
        published = kafka_producer.send_event(obs, topic=KAFKA_TOPIC)
        if published:
            success_count += 1
            logger.info(
                " [Kafka -> %s] Road: %-15s | Speed: %4.1f km/h | Free: %4.1f km/h | Ratio: %s | Status: %s | Source: %s",
                KAFKA_TOPIC,
                obs.get("road_id"),
                obs.get("current_speed") or 0.0,
                obs.get("free_flow_speed") or 0.0,
                f"{obs.get('congestion_ratio'):.2f}" if obs.get("congestion_ratio") is not None else "N/A",
                obs.get("congestion_level", "unknown").upper(),
                obs.get("data_source")
            )
        else:
            logger.warning(
                " [Kafka FAILED] Could not publish event for road '%s' to topic '%s'",
                obs.get("road_id"), KAFKA_TOPIC
            )

    logger.info("Cycle finished: %d / %d events published to Kafka.\n", success_count, len(observations))
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Pune Traffic Intelligence Data Producer")
    parser.add_argument(
        "--mode",
        choices=["auto", "tomtom", "demo"],
        default=DATA_MODE if DATA_MODE in ["tomtom", "demo"] else "auto",
        help="Data collection mode (auto detects API key, tomtom uses live API, demo uses synthetic stream)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single collection cycle and exit (useful for testing)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuously at specified interval"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=COLLECTION_INTERVAL_SECONDS,
        help="Interval in seconds between collection cycles (default: 300s = 5m)"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable appending raw observations to data/raw/traffic_raw_events.csv"
    )
    args = parser.parse_args()

    # Determine active mode
    client = TomTomTrafficClient(api_key=TOMTOM_API_KEY)
    if args.mode == "auto":
        mode = "tomtom" if client.is_api_key_available() else "demo"
    else:
        mode = args.mode

    logger.info("=" * 70)
    logger.info(" Pune Traffic Intelligence - Event Producer")
    logger.info(" Active Mode:       %s", mode.upper())
    logger.info(" Kafka Bootstrap:   %s", KAFKA_BOOTSTRAP_SERVERS)
    logger.info(" Kafka Topic:       %s", KAFKA_TOPIC)
    logger.info(" Polling Interval:  %d seconds", args.interval)
    logger.info(" Save to CSV:       %s", not args.no_csv)
    logger.info(" Configured Roads:  %d Pune points", len(PUNE_ROAD_POINTS))
    if mode == "tomtom":
        logger.info(" TomTom API Key:    %s", mask_key(TOMTOM_API_KEY))
    logger.info("=" * 70)

    # Initialize Kafka Producer
    kafka_producer = TrafficKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    connected = kafka_producer.connect()
    if not connected:
        logger.error(
            "\n[CRITICAL] Kafka Broker is not reachable at '%s'!\n"
            "Please ensure the Kafka container is running:\n"
            "   docker compose up -d\n"
            "Then verify topic exists:\n"
            "   python scripts/create_topic.py\n",
            KAFKA_BOOTSTRAP_SERVERS
        )
        # We don't crash outright if running in one-shot test mode, but inform clearly
        if not args.once:
            sys.exit(1)

    try:
        # Run one-shot or continuous
        if args.once:
            run_collection_cycle(client, kafka_producer, mode=mode, save_csv=not args.no_csv)
        else:
            logger.info("Entering continuous streaming loop. Press Ctrl+C to stop.")
            while True:
                run_collection_cycle(client, kafka_producer, mode=mode, save_csv=not args.no_csv)
                logger.info("Sleeping for %d seconds before next cycle...", args.interval)
                time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Producer stopped by user (Ctrl+C).")
    finally:
        kafka_producer.close()


if __name__ == "__main__":
    main()
