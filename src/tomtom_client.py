"""
TomTom Traffic Flow Segment API Client.
Interacts with the official TomTom Traffic Flow API to retrieve real-time
traffic flow segment data for defined coordinates in Pune.
Includes defensive error handling, key masking, and demo-mode fallback.
"""

import logging
import random
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import requests

from config.settings import TOMTOM_FLOW_ENDPOINT, TOMTOM_API_KEY, PUNE_ROAD_POINTS
from src.congestion import enrich_record_with_congestion

# Configure logger
logger = logging.getLogger(__name__)


def mask_key(key: str) -> str:
    """Masks API key for safe logging (e.g., 'ab12****89yz')."""
    if not key or len(key) < 8:
        return "[REDACTED/EMPTY]"
    return f"{key[:4]}****{key[-4:]}"


class TomTomTrafficClient:
    """
    Client for TomTom Flow Segment Data API.
    Endpoint: https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json
    """

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 10):
        self.api_key = api_key if api_key is not None else TOMTOM_API_KEY
        self.timeout = timeout_seconds

        if not self.api_key:
            logger.warning("No TomTom API key provided. Client will operate in demo fallback mode.")
        else:
            logger.info("Initialized TomTomTrafficClient with API key: %s", mask_key(self.api_key))

    def is_api_key_available(self) -> bool:
        """Returns True if a non-empty, non-placeholder API key is set."""
        return bool(self.api_key and "your_tomtom" not in self.api_key.lower())

    def fetch_flow_segment(
        self,
        latitude: float,
        longitude: float,
        unit: str = "KMPH"
    ) -> Optional[Dict[str, Any]]:
        """
        Calls TomTom Flow Segment Data API for given coordinates.
        Handles HTTP errors, timeouts, rate limits, and malformed responses.
        """
        if not self.is_api_key_available():
            logger.debug("API key not configured. Cannot call TomTom API.")
            return None

        params = {
            "point": f"{latitude},{longitude}",
            "unit": unit,
            "key": self.api_key
        }

        try:
            response = requests.get(
                TOMTOM_FLOW_ENDPOINT,
                params=params,
                timeout=self.timeout
            )

            # Handle Rate Limiting (HTTP 429)
            if response.status_code == 429:
                logger.warning("TomTom API rate limit exceeded (HTTP 429). Retry later.")
                return None

            # Handle Authentication / Authorization (HTTP 401 / 403)
            if response.status_code in (401, 403):
                logger.error(
                    "TomTom API authentication failed (HTTP %d). "
                    "Please verify your TOMTOM_API_KEY in .env.", response.status_code
                )
                return None

            # Handle other HTTP errors
            response.raise_for_status()

            data = response.json()
            return data.get("flowSegmentData")

        except requests.exceptions.Timeout:
            logger.warning("Request timed out for point (%s, %s)", latitude, longitude)
            return None
        except requests.exceptions.ConnectionError:
            logger.warning("Network connection error connecting to TomTom API")
            return None
        except requests.exceptions.HTTPError as err:
            logger.error("HTTP error from TomTom API: %s", err)
            return None
        except ValueError:
            logger.error("Failed to parse JSON response from TomTom API")
            return None
        except Exception as e:
            logger.error("Unexpected error fetching traffic data: %s", e)
            return None

    def fetch_pune_snapshot(
        self,
        road_points: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches live flow observations for all configured Pune road points.
        Returns a list of standardized traffic observation dictionaries.
        """
        points = road_points or PUNE_ROAD_POINTS
        observations = []
        now_utc = datetime.now(timezone.utc).isoformat()

        for pt in points:
            road_id = pt["road_id"]
            lat = pt["latitude"]
            lon = pt["longitude"]

            flow_data = self.fetch_flow_segment(lat, lon)

            if flow_data:
                # Successfully received response from TomTom
                cur_speed = flow_data.get("currentSpeed")
                free_speed = flow_data.get("freeFlowSpeed")
                cur_time = flow_data.get("currentTravelTime")
                free_time = flow_data.get("freeFlowTravelTime")
                confidence = flow_data.get("confidence")
                road_closure = flow_data.get("roadClosure", False)
                frc = flow_data.get("frc")

                record = {
                    "timestamp": now_utc,
                    "road_id": road_id,
                    "road_name": pt.get("name", road_id),
                    "latitude": lat,
                    "longitude": lon,
                    "frc": frc,
                    "current_speed": float(cur_speed) if cur_speed is not None else None,
                    "free_flow_speed": float(free_speed) if free_speed is not None else None,
                    "current_travel_time": int(cur_time) if cur_time is not None else None,
                    "free_flow_travel_time": int(free_time) if free_time is not None else None,
                    "confidence": float(confidence) if confidence is not None else 1.0,
                    "road_closure": bool(road_closure),
                    "data_source": "tomtom"
                }
                enrich_record_with_congestion(record)
                observations.append(record)
            else:
                logger.debug("No live flow data returned for %s", road_id)

        return observations

    @staticmethod
    def generate_demo_observation(
        pt: Dict[str, Any],
        timestamp_utc: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a realistic synthetic observation for a road point.
        Takes time-of-day traffic patterns into account:
        - Morning peak (8:30 - 11:30 AM IST = 03:00 - 06:00 UTC)
        - Evening peak (5:30 - 9:00 PM IST = 12:00 - 15:30 UTC)
        - Normal/Off-peak (other times)
        """
        if timestamp_utc is None:
            now_dt = datetime.now(timezone.utc)
            timestamp_utc = now_dt.isoformat()
        else:
            try:
                now_dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
            except Exception:
                now_dt = datetime.now(timezone.utc)

        road_id = pt["road_id"]
        lat = pt["latitude"]
        lon = pt["longitude"]
        baseline_free_flow = pt.get("typical_free_flow_speed", 50.0)

        # Free flow speed has slight variations
        free_flow_speed = round(baseline_free_flow + random.uniform(-2, 3), 1)

        # Determine time-of-day factor (in IST = UTC + 5.5 hours)
        utc_hour = now_dt.hour + (now_dt.minute / 60.0)
        ist_hour = (utc_hour + 5.5) % 24

        # Peak hours modeling
        is_morning_peak = 8.5 <= ist_hour <= 11.5
        is_evening_peak = 17.5 <= ist_hour <= 21.0
        is_night = 23.0 <= ist_hour or ist_hour < 6.0

        # Determine speed multiplier
        if is_morning_peak:
            # Heavy traffic: 25% to 65% of free flow speed
            speed_mult = random.uniform(0.25, 0.65)
        elif is_evening_peak:
            # Heavy traffic: 20% to 60% of free flow speed
            speed_mult = random.uniform(0.20, 0.60)
        elif is_night:
            # Open road: 85% to 100% of free flow speed
            speed_mult = random.uniform(0.85, 1.0)
        else:
            # Daytime off-peak: 60% to 85% of free flow speed
            speed_mult = random.uniform(0.60, 0.85)

        # 2% chance of road closure simulation
        road_closure = random.random() < 0.02
        if road_closure:
            current_speed = 0.0
            confidence = 1.0
        else:
            current_speed = round(max(5.0, free_flow_speed * speed_mult), 1)
            confidence = round(random.uniform(0.75, 1.0), 2)

        # Calculate travel times (distance approx = free_flow_speed * 120s / 3600 = free_flow_speed / 30 km)
        distance_km = round(free_flow_speed / 30.0, 2)
        free_travel_time = int((distance_km / max(free_flow_speed, 1.0)) * 3600)
        if current_speed > 0:
            cur_travel_time = int((distance_km / current_speed) * 3600)
        else:
            cur_travel_time = 9999

        record = {
            "timestamp": timestamp_utc,
            "road_id": road_id,
            "road_name": pt.get("name", road_id),
            "latitude": lat,
            "longitude": lon,
            "frc": "FRC1",
            "current_speed": current_speed,
            "free_flow_speed": free_flow_speed,
            "current_travel_time": cur_travel_time,
            "free_flow_travel_time": free_travel_time,
            "confidence": confidence,
            "road_closure": road_closure,
            "data_source": "synthetic_demo"
        }
        enrich_record_with_congestion(record)
        return record

    def generate_demo_snapshot(
        self,
        road_points: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Generates synthetic snapshot for all road points."""
        points = road_points or PUNE_ROAD_POINTS
        now_utc = datetime.now(timezone.utc).isoformat()
        return [self.generate_demo_observation(pt, timestamp_utc=now_utc) for pt in points]
