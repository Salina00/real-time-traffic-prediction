# IoT-Enabled Real-Time Traffic Intelligence and Congestion Prediction System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-3.7.0_(KRaft)-red.svg)](https://kafka.apache.org/)
[![Spark](https://img.shields.io/badge/Apache_Spark-3.5.1-orange.svg)](https://spark.apache.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Big Data streaming, IoT event processing, and Machine Learning platform for Pune City road corridors. Built for **Computer Engineering College Mini-Projects** using **Apache Kafka**, **Apache Spark Structured Streaming**, **TomTom Traffic Flow API**, **Scikit-Learn**, and **Streamlit**.

---

## 🏛️ Pipeline Architecture

```
                                  ┌────────────────────────┐
                                  │   TomTom Traffic API   │
                                  │   (or Synthetic Demo)  │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │ Python Traffic Producer│
                                  │     (producer.py)      │
                                  └───────────┬────────────┘
                                              │ (JSON Events)
                                              ▼
                             ┌──────────────────────────────────┐
                             │ Apache Kafka Broker (KRaft Mode) │
                             │     Topic: pune-traffic-raw      │
                             └────────────────┬─────────────────┘
                                              │
                                              ▼
                             ┌──────────────────────────────────┐
                             │ Apache Spark Structured Streaming│
                             │      (spark_streaming.py)        │
                             │  - Strict Schema Validation      │
                             │  - Congestion Ratio Calculation  │
                             │  - Speed Level Classification    │
                             └───────────┬───────────┬──────────┘
                                         │           │
                     ┌───────────────────┘           └───────────────────┐
                     ▼                                                   ▼
         ┌───────────────────────┐                           ┌───────────────────────┐
         │ Persistent Parquet    │                           │ Real-Time Stream CSV  │
         │ data/processed/stream │                           │ data/processed/*.csv  │
         └───────────┬───────────┘                           └───────────┬───────────┘
                     │                                                   │
                     ▼                                                   ▼
         ┌───────────────────────┐                           ┌───────────────────────┐
         │ 15-Min Prediction ML  │                           │  Streamlit Command    │
         │   (train_model.py)    │                           │    Center (app.py)    │
         └───────────────────────┘                           └───────────────────────┘
```

---

## ⚙️ Technology Stack & Compatible Versions

| Component | Selected Version | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 | PowerShell & VS Code |
| **Python** | 3.11.x (or 3.10+) | Pinned dependencies in `.venv` |
| **Apache Kafka** | 3.7.0 | Single-node local broker running in **KRaft mode** (no ZooKeeper required) |
| **Apache Spark** | 3.5.1 | PySpark Structured Streaming |
| **Kafka Connector** | `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1` | Scala 2.12 connector for Spark 3.5 |
| **Java Runtime** | OpenJDK 17 or 21 (Temurin) | Required for Spark execution |
| **Traffic API** | TomTom Flow Segment Data v4 | REST API with coordinate-based queries |
| **Machine Learning**| scikit-learn 1.5.x | Unsupervised K-Means clustering & supervised RandomForestClassifier |
| **Dashboard** | Streamlit + Plotly Express | Navy-themed interactive real-time dashboard |

---

## 📁 Project Structure

```
traffic-intelligence/
├── app.py                      # Streamlit real-time interactive dashboard (with K-Means section)
├── producer.py                 # Traffic event collector (TomTom API or demo stream)
├── spark_streaming.py          # PySpark Structured Streaming consumer from Kafka
├── train_model.py              # Supervised ML: 15-min horizon prediction (Random Forest)
├── requirements.txt            # Pinned Python package dependencies
├── .env.example                # Template for configuration & credentials
├── .env                        # Local active environment configuration
├── .gitignore                  # Git ignore rules for logs, venv, and checkpoints
├── docker-compose.yml          # Kafka KRaft container + auto-topic-init
├── Dockerfile.spark            # Containerized Spark execution alternative
├── README.md                   # Complete documentation and execution guide
├── config/
│   ├── __init__.py
│   └── settings.py             # Configurable Pune roads, coordinates, thresholds & K-Means config
├── src/
│   ├── __init__.py
│   ├── clustering.py           # Unsupervised K-Means clustering & dynamic label interpretation
│   ├── tomtom_client.py        # TomTom Traffic API client & demo generator
│   ├── congestion.py           # Speed ratio math & classification logic
│   ├── kafka_producer.py       # Resilient Kafka producer wrapper
│   └── data_utils.py           # Dataset cleaning, validation & target shifts
├── scripts/
│   ├── __init__.py
│   ├── create_topic.py         # Idempotent Kafka topic creator & inspector
│   ├── generate_demo_data.py   # Synthetic historical dataset generator
│   └── check_environment.py   # Diagnostic script verifying system readiness
├── data/
│   ├── raw/                    # Raw event append logs (CSV)
│   ├── processed/              # Spark streaming output (Parquet & CSV) & clustered CSV
│   └── demo/                   # Labelled synthetic demonstration datasets
├── models/                     # Saved ML models, encoders & K-Means artifacts
├── logs/                       # Application execution logs
└── tests/
    ├── __init__.py
    ├── test_clustering.py      # Tests for K-Means preprocessing, scaling, dynamic labels & fallback
    ├── test_congestion.py      # Tests for thresholds, closures & zero-speed
    └── test_data_validation.py # Tests for duplicates, formatting & tolerance
```

---

## 🚦 Congestion Classification Logic

The system computes the speed ratio:
$$\text{congestion\_ratio} = \frac{\text{current\_speed}}{\text{free\_flow\_speed}}$$

| Condition | Congestion Level | Action / Badge |
| :--- | :--- | :--- |
| `road_closure == True` | `closed` | Physical closure reported by API/sensors |
| `ratio >= 0.70` | `low` | Free-flowing traffic |
| `0.40 <= ratio < 0.70` | `moderate` | Approaching rush-hour congestion |
| `ratio < 0.40` | `high` | Severe bottleneck / heavy delay |
| Missing / Free-flow $\le 0$ | `unknown` | Invalid or unavailable sensor reading |

*Note: Thresholds are configurable via `config/settings.py`.*

---

## 🧩 Unsupervised Machine Learning: K-Means Traffic Clustering

In addition to rule-based classification and supervised forward-looking forecasting, the system incorporates **K-Means clustering** (`src/clustering.py`) as an unsupervised data mining technique.

### 1. Why K-Means is Appropriate for Traffic Intelligence
- **Unsupervised Pattern Discovery**: Unlike supervised models which require ground-truth target labels, K-Means discovers intrinsic clusters in multivariate traffic flow data without human annotation.
- **Dynamic Road Regimes**: Urban corridors have varying speed limits and geometric capacities. By clustering multidimensional features, K-Means uncovers natural operational states (e.g. smooth flow, creeping congestion, severe bottlenecks) directly from current traffic behavior.
- **Separate from Future Prediction**:
  - **K-Means Clustering (Unsupervised)**: Explores and groups *current* or historical traffic states across road segments. It does **not** predict future congestion.
  - **Random Forest Forecaster (Supervised)**: Trains on historical target lags to predict congestion states ~15 minutes into the future.

### 2. Features Selected
Clustering uses three continuous numerical features:
1. `current_speed` (km/h): Observed vehicle speed from TomTom API or sensors.
2. `free_flow_speed` (km/h): Uncongested baseline road speed.
3. `congestion_ratio`: Ratio of current speed to free-flow speed ($\frac{\text{current\_speed}}{\text{free\_flow\_speed}}$).

Categorical columns (such as `road_id`, `road_name`, or timestamps) are excluded from the mathematical feature matrix to prevent artificial distance distortions.

### 3. Preprocessing & Feature Scaling
- **Data Hygiene**: Filters out rows with missing (`null`), non-finite (`inf`, `-inf`), or negative speeds.
- **Division-by-Zero Protection**: Observations with $\text{free\_flow\_speed} \le 0$ are screened out before computing ratios.
- **Deduplication**: Duplicate snapshots for the same road segment and timestamp are removed.
- **StandardScaler**: Because speed is measured in km/h (typically 0–80) while `congestion_ratio` is dimensionless (typically 0.0–1.0), feeding raw numbers would cause the speed scale to overpower the congestion ratio. `StandardScaler` normalizes each feature to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$), ensuring equal geometric contribution during Euclidean distance calculation.
- **Reproducibility**: Initialized with fixed `random_state=42` and `n_init=10`.

### 4. Cluster Assignment & Dynamic Interpretation
1. **Model Fitting**: Partitions the observations into $K$ clusters (default $K=3$) by minimizing within-cluster sum of squares (inertia).
2. **Centroid Inspection**: Rather than assuming that cluster ID 0 or 1 maps to a specific severity, the system inspects the unscaled cluster centroids:
   $$\text{Mean Congestion Ratio} = \frac{1}{|C_k|} \sum_{i \in C_k} \text{congestion\_ratio}_i$$
   $$\text{Mean Current Speed} = \frac{1}{|C_k|} \sum_{i \in C_k} \text{current\_speed}_i$$
3. **Semantic Mapping ($K=3$)**:
   - **Heavy Congestion**: The cluster with the **lowest** mean congestion ratio (and lowest average speed).
   - **Moderate Congestion**: The cluster with the **intermediate** congestion ratio.
   - **Light Congestion**: The cluster with the **highest** congestion ratio (traffic operating near free-flow capacity).

### 5. Running K-Means from CLI & Viewing in Streamlit
- **CLI Batch Execution**:
  ```powershell
  # Cluster currently available stream or raw CSV data
  python src/clustering.py --k 3
  
  # Or with Python module syntax
  python -m src.clustering --k 3
  ```
  This outputs learned centroid statistics in the console, saves the model to `models/kmeans_traffic_model.joblib`, and exports annotated records to `data/processed/clustered_traffic_latest.csv`.

- **Streamlit Interactive Command Center**:
  ```powershell
  streamlit run app.py
  ```
  Navigate to the **“🧩 Traffic Clustering — K-Means”** section to view:
  - Configurable $K$ cluster slider (2 to 5).
  - Scope toggle: latest road corridor snapshot vs full active stream.
  - KPI metric cards showing cluster counts, percentages, and data source badges.
  - An interactive Plotly scatter plot (Current Speed vs Congestion Ratio) with distinct color coding and overlayed cluster centroids (star markers).
  - Learned centroid table in physical units (km/h).
  - Full road-level clustered observation table.

### 6. Known Limitations
- **Sample Size Requirement**: K-Means requires at least $K$ valid observations to fit (and ideally $\ge 6$ observations for stable centroids). On small samples, centroid boundaries can be influenced by local anomalies.
- **Batch / Offline Execution**: K-Means operates on historical batches or snapshots; it is not executed inside Spark's distributed micro-batch engine to avoid streaming driver latency.
- **Validation on Real vs Demo Data**: While the clustering algorithm works identically on real TomTom API data and synthetic demo data, production deployments require accumulating continuous 24/7 observations across seasons and weather conditions.

---

## 🚀 Step-by-Step Windows & VS Code Guide

### 1. Prerequisites Check
Open PowerShell in the project directory and verify tool readiness:
```powershell
python scripts/check_environment.py
```

### 2. Activate Virtual Environment
If `.venv` is not active, activate it:
```powershell
# PowerShell:
.\.venv\Scripts\Activate.ps1

# If execution policy blocks scripts, enable for current process:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Configure `.env`
Ensure `.env` exists (copied from `.env.example`):
```ini
TOMTOM_API_KEY=your_key_here
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=pune-traffic-raw
COLLECTION_INTERVAL_SECONDS=300
DATA_MODE=demo
```
- If you have a TomTom API Key, set `DATA_MODE=tomtom` and paste your key.
- If you do not have a key, keep `DATA_MODE=demo`. The entire pipeline will function identically using realistic synthetic stream events.

### 4. Start Kafka Broker (Docker)
In your terminal, start the Apache Kafka KRaft broker:
```powershell
docker compose up -d
```
Verify the container status:
```powershell
docker compose ps
```
Verify the topic `pune-traffic-raw` is ready:
```powershell
python scripts/create_topic.py
```

### 5. Generate Demo Data & Train Baseline ML Model
Before starting the live streams, generate 3 days of historical training data and train the 15-minute forecasting model:
```powershell
python scripts/generate_demo_data.py --days 3
python train_model.py
```

### 6. Run the End-to-End Live Pipeline

Open **4 separate PowerShell terminals** in VS Code:

#### 🖥️ Terminal 1: Apache Spark Structured Streaming
Reads streaming JSON events from Kafka, validates them, and writes to Parquet and CSV:
```powershell
.\.venv\Scripts\Activate.ps1
python spark_streaming.py --console
```

#### 🖥️ Terminal 2: Traffic Event Producer
Collects data from TomTom API (or demo stream) and publishes to `pune-traffic-raw`:
```powershell
.\.venv\Scripts\Activate.ps1
# Continuous streaming (5-min interval):
python producer.py --continuous --interval 60

# Or run a single test cycle:
python producer.py --once
```

#### 🖥️ Terminal 3: Streamlit Real-Time Dashboard
Launches the command center with real-time maps, Plotly charts, and ML forecasting:
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```
*The dashboard will automatically open in your default browser at `http://localhost:8501`.*

#### 🖥️ Terminal 4: Verification & Inspection
Use this terminal to test components, run tests, or inspect Kafka topic messages:
```powershell
.\.venv\Scripts\Activate.ps1
# Run unit tests
pytest -v tests/
```

---

## 🧪 Testing & Verification

Run the full automated test suite:
```powershell
pytest -v tests/
```
All 22 unit tests test:
- **K-Means Clustering (`test_clustering.py`)**:
  - Feature extraction and validation
  - Missing, invalid, and non-finite value exclusion
  - Zero and negative free-flow speed protection (prevents division by zero)
  - Insufficient sample error handling and graceful fallbacks
  - Cluster assignment and dynamic interpretation (Heavy, Moderate, Light) based on centroid ratio/speed
  - Reproducibility with fixed random seed (`random_state=42`)
  - Cluster statistics and centroid calculation in physical units
  - Model serialization and deserialization via joblib
  - Dashboard helper fallback on empty, corrupted, or missing data
- **Congestion Classification (`test_congestion.py`)**:
  - Congestion threshold boundaries (`low`, `moderate`, `high`)
  - Road closure override behavior
  - Missing, negative, and zero free-flow speed edge cases
  - Custom threshold support
- **Data Validation & Pipelines (`test_data_validation.py`)**:
  - Required field validation and ISO 8601 timestamp parsing
  - Malformed event rejection
  - Duplicate record removal
  - Tolerance-based future target generation (15-min horizon without lookahead leakage)
  - Chronological train/test splitting integrity

---

## 🛑 How to Stop Services

To gracefully terminate the streaming components:
1. In the Producer, Spark, and Streamlit terminals, press `Ctrl + C`.
2. Stop the Kafka Docker broker:
   ```powershell
   docker compose down
   ```

---

## 🛠️ Common Errors & Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| `NoBrokersAvailable` | Kafka container is not running | Run `docker compose up -d` and wait ~10 seconds. Check `docker compose logs kafka`. |
| `ExecutionPolicy` error in PowerShell | Windows blocks activate script | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then activate `.venv`. |
| `Java not found` | OpenJDK is not in PATH | Install OpenJDK 17 or 21 (Temurin) or run Spark in Docker: `docker compose run --rm spark-streaming`. |
| `TomTom HTTP 403 / 401` | Invalid API key | Verify key in `.env` or set `DATA_MODE=demo`. |
| Spark Kafka connector download fails | Firewall blocking Maven Central | Spark downloads `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1` on first run. Ensure internet access is enabled. |
