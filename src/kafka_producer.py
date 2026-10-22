"""
Kafka Producer wrapper for Pune Traffic Events.
Handles connection health check, JSON serialization, key partitioning,
and error handling.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC

logger = logging.getLogger(__name__)


class TrafficKafkaProducer:
    """
    Manages connection and publishing to Kafka broker.
    """

    def __init__(self, bootstrap_servers: Optional[str] = None):
        self.bootstrap_servers = bootstrap_servers or KAFKA_BOOTSTRAP_SERVERS
        self._producer: Optional[KafkaProducer] = None

    def connect(self) -> bool:
        """
        Attempts to connect to Kafka broker.
        Returns True if successful, False otherwise.
        """
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
                request_timeout_ms=10000,
                api_version_auto_timeout_ms=10000,
            )
            logger.info("Connected to Kafka broker at %s", self.bootstrap_servers)
            return True
        except NoBrokersAvailable:
            logger.error(
                "No Kafka broker available at '%s'. "
                "Ensure Kafka container is running via: docker compose up -d",
                self.bootstrap_servers
            )
            self._producer = None
            return False
        except Exception as e:
            logger.error("Failed to initialize KafkaProducer: %s", e)
            self._producer = None
            return False

    @property
    def is_connected(self) -> bool:
        return self._producer is not None

    def send_event(self, event: Dict[str, Any], topic: Optional[str] = None) -> bool:
        """
        Sends a single traffic event dictionary to Kafka topic.
        Partition key is set to road_id for road-level ordering.
        """
        if not self.is_connected and not self.connect():
            return False

        target_topic = topic or KAFKA_TOPIC
        road_id = event.get("road_id", "unknown")

        try:
            future = self._producer.send(
                target_topic,
                key=road_id,
                value=event
            )
            # Block briefly to ensure event is sent
            record_metadata = future.get(timeout=10.0)
            logger.debug(
                "Sent event for road '%s' to %s [partition %d, offset %d]",
                road_id, target_topic, record_metadata.partition, record_metadata.offset
            )
            return True
        except KafkaError as ke:
            logger.error("Kafka error while sending event for road '%s': %s", road_id, ke)
            return False
        except Exception as e:
            logger.error("Unexpected error sending event to Kafka: %s", e)
            return False

    def send_batch(self, events: List[Dict[str, Any]], topic: Optional[str] = None) -> int:
        """
        Sends a list of events to Kafka.
        Returns count of successfully delivered events.
        """
        success_count = 0
        for ev in events:
            if self.send_event(ev, topic=topic):
                success_count += 1
        if self._producer:
            self._producer.flush()
        return success_count

    def close(self):
        """Flushes and closes Kafka producer."""
        if self._producer:
            try:
                self._producer.flush(timeout=5.0)
                self._producer.close(timeout=5.0)
                logger.info("Kafka producer closed gracefully.")
            except Exception as e:
                logger.warning("Error closing Kafka producer: %s", e)
            finally:
                self._producer = None
