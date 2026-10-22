"""
Configuration module for IoT-Enabled Real-Time Traffic Intelligence System.
Centralizes environment variables, road network coordinates, thresholds,
and Kafka / Spark settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Suppress joblib loky warning on Windows if wmic/powershell query fails
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# Security & Credentials
# ==============================================================================
TOMTOM_API_KEY = os.getenv("TOMTOM_API_KEY", "").strip()

# ==============================================================================
# Mode & Operational Settings
# ==============================================================================
# "tomtom" for live API calls, "demo" for realistic synthetic events
DATA_MODE = os.getenv("DATA_MODE", "demo").strip().lower()

# Producer collection interval in seconds (default: 300s = 5 minutes)
COLLECTION_INTERVAL_SECONDS = int(os.getenv("COLLECTION_INTERVAL_SECONDS", "300"))

# ==============================================================================
# Kafka Settings
# ==============================================================================
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "pune-traffic-raw")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "pune-traffic-spark-group")

# ==============================================================================
# Storage Paths
# ==============================================================================
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = BASE_DIR / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = BASE_DIR / os.getenv("PROCESSED_DATA_DIR", "data/processed")
DEMO_DATA_DIR = BASE_DIR / os.getenv("DEMO_DATA_DIR", "data/demo")
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SPARK_CHECKPOINT_DIR = BASE_DIR / os.getenv("SPARK_CHECKPOINT_DIR", "data/processed/checkpoints")

# Ensure required directories exist
def init_directories():
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, DEMO_DATA_DIR, MODELS_DIR, LOGS_DIR, SPARK_CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

init_directories()

# ==============================================================================
# Configurable Pune Road Points
# Approximate coordinates of major traffic junctions & corridors in Pune
# ==============================================================================
PUNE_ROAD_POINTS = [
    {
        "road_id": "pune_central",
        "name": "Pune Central (FC Road / Deccan)",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "typical_free_flow_speed": 45.0,
    },
    {
        "road_id": "shivajinagar",
        "name": "Shivajinagar Junction",
        "latitude": 18.5308,
        "longitude": 73.8475,
        "typical_free_flow_speed": 50.0,
    },
    {
        "road_id": "swargate",
        "name": "Swargate Bus Terminal & Flyover",
        "latitude": 18.5018,
        "longitude": 73.8636,
        "typical_free_flow_speed": 40.0,
    },
    {
        "road_id": "hinjawadi",
        "name": "Hinjawadi IT Park Phase 1",
        "latitude": 18.5913,
        "longitude": 73.7389,
        "typical_free_flow_speed": 60.0,
    },
    {
        "road_id": "hadapsar",
        "name": "Hadapsar Magarpatta Bypass",
        "latitude": 18.5089,
        "longitude": 73.9259,
        "typical_free_flow_speed": 55.0,
    },
    {
        "road_id": "viman_nagar",
        "name": "Viman Nagar (Pune Airport Road)",
        "latitude": 18.5679,
        "longitude": 73.9143,
        "typical_free_flow_speed": 50.0,
    },
    {
        "road_id": "kothrud",
        "name": "Kothrud (Paud Road)",
        "latitude": 18.5074,
        "longitude": 73.8077,
        "typical_free_flow_speed": 45.0,
    },
    {
        "road_id": "katraj",
        "name": "Katraj Tunnel Highway",
        "latitude": 18.4575,
        "longitude": 73.8677,
        "typical_free_flow_speed": 65.0,
    },
]

# Quick lookup by road_id
ROAD_POINTS_MAP = {pt["road_id"]: pt for pt in PUNE_ROAD_POINTS}

# ==============================================================================
# Congestion Thresholds
# ==============================================================================
# ratio = current_speed / free_flow_speed
CONGESTION_THRESHOLDS = {
    "low_min": 0.70,       # ratio >= 0.70 -> low
    "moderate_min": 0.40,  # 0.40 <= ratio < 0.70 -> moderate
    # ratio < 0.40 -> high
}

# ==============================================================================
# K-Means Clustering Settings (Unsupervised Data Mining)
# ==============================================================================
KMEANS_DEFAULT_CLUSTERS = int(os.getenv("KMEANS_DEFAULT_CLUSTERS", "3"))
KMEANS_RANDOM_STATE = int(os.getenv("KMEANS_RANDOM_STATE", "42"))
KMEANS_MIN_OBSERVATIONS = int(os.getenv("KMEANS_MIN_OBSERVATIONS", "6"))
KMEANS_FEATURE_COLS = ["current_speed", "free_flow_speed", "congestion_ratio"]
KMEANS_MODEL_PATH = MODELS_DIR / "kmeans_traffic_model.joblib"

# ==============================================================================
# Isolation Forest Settings (Unsupervised Anomaly Detection)
# ==============================================================================
# Contamination factor: expected proportion of outliers/anomalies in urban traffic (~5%)
ISOLATION_FOREST_CONTAMINATION = float(os.getenv("ISOLATION_FOREST_CONTAMINATION", "0.05"))
ISOLATION_FOREST_RANDOM_STATE = int(os.getenv("ISOLATION_FOREST_RANDOM_STATE", "42"))
ISOLATION_FOREST_MODEL_PATH = MODELS_DIR / "isolation_forest_model.joblib"
ISOLATION_FOREST_METADATA_PATH = MODELS_DIR / "anomaly_metadata.json"
ISOLATION_FOREST_FEATURE_COLS = ["current_speed", "free_flow_speed", "congestion_ratio", "speed_deficit"]

# ==============================================================================
# Random Forest Congestion Prediction Settings (Supervised Forecasting)
# ==============================================================================
CONGESTION_MODEL_PATH = MODELS_DIR / "congestion_model.joblib"
ROAD_ENCODER_PATH = MODELS_DIR / "road_encoder.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# TomTom Flow Segment Data API Endpoint
TOMTOM_FLOW_ENDPOINT = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
