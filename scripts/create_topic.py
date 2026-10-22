"""
Script to create or verify the Apache Kafka topic: 'pune-traffic-raw'.
Ensures idempotent execution (will not fail if topic already exists)
and prints topic partition and replication metadata.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable, KafkaError
from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_or_verify_topic(
    bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
    topic_name: str = KAFKA_TOPIC,
    partitions: int = 1,
    replication_factor: int = 1
) -> bool:
    """
    Connects to Kafka broker and ensures topic exists.
    """
    logger.info("Connecting to Kafka broker at: %s", bootstrap_servers)
    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            client_id="pune-traffic-admin",
            request_timeout_ms=10000
        )
    except NoBrokersAvailable:
        logger.error(
            "Could not connect to Kafka broker at '%s'.\n"
            "--> Is Kafka running?\n"
            "--> Run: docker compose up -d\n"
            "--> Check container status: docker compose ps",
            bootstrap_servers
        )
        return False
    except Exception as e:
        logger.error("Failed to initialize KafkaAdminClient: %s", e)
        return False

    try:
        existing_topics = admin_client.list_topics()
        if topic_name in existing_topics:
            logger.info("Topic '%s' already exists in Kafka.", topic_name)
        else:
            logger.info("Creating topic '%s' (partitions=%d, replication_factor=%d)...",
                        topic_name, partitions, replication_factor)
            new_topic = NewTopic(
                name=topic_name,
                num_partitions=partitions,
                replication_factor=replication_factor
            )
            try:
                admin_client.create_topics(new_topics=[new_topic], validate_only=False)
                logger.info("Successfully created topic '%s'.", topic_name)
            except TopicAlreadyExistsError:
                logger.info("Topic '%s' was just created by another client.", topic_name)

        # Inspect topic metadata
        cluster_metadata = admin_client.describe_topics([topic_name])
        for topic_meta in cluster_metadata:
            t_name = topic_meta.get("topic")
            parts = topic_meta.get("partitions", [])
            logger.info(
                "Topic Metadata -> Name: '%s', Partitions: %d, ErrorCode: %s",
                t_name, len(parts), topic_meta.get("error_code")
            )
            for p in parts:
                logger.info(
                    "   Partition %d -> Leader: %s, Replicas: %s, ISR: %s",
                    p.get("partition"), p.get("leader"), p.get("replicas"), p.get("isr")
                )

        admin_client.close()
        return True

    except KafkaError as ke:
        logger.error("Kafka error while checking/creating topic: %s", ke)
        return False
    except Exception as e:
        logger.error("Unexpected error in topic creation: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="Create or verify Kafka topic for Pune traffic.")
    parser.add_argument("--bootstrap-servers", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--topic", default=KAFKA_TOPIC, help="Topic name to create")
    parser.add_argument("--partitions", type=int, default=1, help="Number of partitions (default: 1)")
    parser.add_argument("--replication-factor", type=int, default=1, help="Replication factor (default: 1)")
    args = parser.parse_args()

    success = create_or_verify_topic(
        bootstrap_servers=args.bootstrap_servers,
        topic_name=args.topic,
        partitions=args.partitions,
        replication_factor=args.replication_factor
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
