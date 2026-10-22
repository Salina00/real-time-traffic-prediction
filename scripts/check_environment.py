"""
Diagnostic tool to check system environment and readiness:
- Python version & virtualenv
- Docker & Docker Compose
- Java Runtime (OpenJDK)
- TomTom API key status (masked)
- Kafka Broker accessibility
- Spark environment & dependencies
- Dataset & Model artifact status
"""

import os
import sys
import socket
import subprocess
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    TOMTOM_API_KEY, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC,
    DATA_MODE, DEMO_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR
)
from src.tomtom_client import mask_key


def print_status(item: str, status: bool, details: str = "", recommendation: str = ""):
    symbol = "[PASS]" if status else "[WARN]"
    print(f" {symbol} {item:<28} : {details}")
    if not status and recommendation:
        print(f"       -> Action: {recommendation}")


def check_python():
    v = sys.version_info
    v_str = f"{v.major}.{v.minor}.{v.micro}"
    is_ok = v.major == 3 and v.minor in (10, 11, 12, 13)
    in_venv = hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    details = f"Python {v_str} (in venv: {in_venv})"
    rec = "Activate or create virtualenv: py -3.11 -m venv .venv" if not in_venv else ""
    print_status("Python Environment", is_ok, details, rec)
    return is_ok


def check_docker():
    try:
        res = subprocess.run(["docker", "info"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        is_ok = (res.returncode == 0)
        details = "Docker daemon is running" if is_ok else "Docker CLI found, daemon unreachable"
        rec = "Start Docker Desktop from Windows Start menu." if not is_ok else ""
    except FileNotFoundError:
        is_ok = False
        details = "Docker CLI not found in PATH"
        rec = "Install Docker Desktop from https://www.docker.com/products/docker-desktop/"
    except Exception as e:
        is_ok = False
        details = f"Error querying Docker: {e}"
        rec = "Start Docker Desktop."
    print_status("Docker Daemon", is_ok, details, rec)
    return is_ok


def check_java():
    try:
        res = subprocess.run(["java", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        output = res.stderr or res.stdout
        first_line = output.splitlines()[0] if output.splitlines() else "Unknown Java version"
        is_ok = (res.returncode == 0)
        details = first_line.strip()
        rec = ""
    except FileNotFoundError:
        is_ok = False
        details = "Java not found in PATH"
        rec = "Install OpenJDK 17 or 21 (Temurin / Eclipse Adoptium) or run Spark in Docker container."
    except Exception as e:
        is_ok = False
        details = f"Error querying Java: {e}"
        rec = "Install OpenJDK."
    print_status("Java Runtime (for Spark)", is_ok, details, rec)
    return is_ok


def check_tomtom_key():
    has_key = bool(TOMTOM_API_KEY and "your_tomtom" not in TOMTOM_API_KEY.lower())
    if has_key:
        details = f"Configured (Key: {mask_key(TOMTOM_API_KEY)})"
        rec = ""
    else:
        details = "Missing or placeholder (Operating in DEMO mode)"
        rec = "Add your key to .env (TOMTOM_API_KEY=...) or continue using DATA_MODE=demo"
    print_status("TomTom API Key", has_key, details, rec)
    return has_key


def check_kafka_broker():
    host, _, port_str = KAFKA_BOOTSTRAP_SERVERS.partition(":")
    port = int(port_str) if port_str else 9092
    host = host if host else "localhost"

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        result = sock.connect_ex((host, port))
        is_ok = (result == 0)
        details = f"Reachable at {host}:{port}" if is_ok else f"Cannot connect to {host}:{port}"
        rec = "Start Kafka container: docker compose up -d" if not is_ok else ""
    except Exception as e:
        is_ok = False
        details = f"Connection error: {e}"
        rec = "Run docker compose up -d"
    finally:
        sock.close()

    print_status("Kafka Broker", is_ok, details, rec)
    return is_ok


def check_data_and_models():
    demo_file = DEMO_DATA_DIR / "pune_traffic_historical_demo.csv"
    has_demo = demo_file.exists()
    details = f"Found demo CSV ({demo_file.name})" if has_demo else "Demo CSV not generated yet"
    rec = "Run: python scripts/generate_demo_data.py" if not has_demo else ""
    print_status("Demo Dataset", has_demo, details, rec)

    model_file = MODELS_DIR / "congestion_model.joblib"
    has_model = model_file.exists()
    details = "Found trained model" if has_model else "Model not trained yet"
    rec = "Run: python train_model.py" if not has_model else ""
    print_status("Trained Model", has_model, details, rec)


def main():
    print("=" * 70)
    print(" IoT-Enabled Real-Time Traffic Intelligence - Environment Diagnostic")
    print("=" * 70)
    check_python()
    check_docker()
    check_java()
    check_tomtom_key()
    check_kafka_broker()
    check_data_and_models()
    print("=" * 70)


if __name__ == "__main__":
    main()
