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
| **Machine Learning**| scikit-learn 1.5.x | Multi-Model ML Architecture: K-Means, Random Forest, Isolation Forest |
| **Dashboard** | Streamlit + Plotly Express | Navy-themed interactive real-time dashboard |

---

## 📁 Project Structure

```
traffic-intelligence/
├── app.py                      # Streamlit interactive command center (K-Means, Random Forest, Isolation Forest)
├── producer.py                 # Traffic event collector (TomTom API or demo stream)
├── spark_streaming.py          # PySpark Structured Streaming consumer from Kafka
├── train_model.py              # Supervised ML entry point (delegates to src/congestion_prediction.py)
├── requirements.txt            # Pinned Python package dependencies
├── .env.example                # Template for configuration & credentials
├── .env                        # Local active environment configuration (excluded from git)
├── .gitignore                  # Git ignore rules for logs, venv, and checkpoints
├── docker-compose.yml          # Kafka KRaft container + auto-topic-init
├── Dockerfile.spark            # Containerized Spark execution alternative
├── README.md                   # Complete documentation and execution guide
├── config/
│   ├── __init__.py
│   └── settings.py             # Configurable Pune roads, thresholds, and ML model parameters
├── src/
│   ├── __init__.py
│   ├── congestion_prediction.py # Supervised Random Forest forecaster (~15-min lookahead)
│   ├── anomaly_detection.py    # Unsupervised Isolation Forest outlier & anomaly detector
│   ├── clustering.py           # Unsupervised K-Means clustering & dynamic label interpretation
│   ├── tomtom_client.py        # TomTom Traffic API client & demo generator
│   ├── congestion.py           # Speed ratio math & classification logic
│   ├── kafka_producer.py       # Resilient Kafka producer wrapper
│   └── data_utils.py           # Dataset cleaning, validation & target shifts
├── scripts/
│   ├── __init__.py
│   ├── train_prediction_model.py # Workflow: Train & evaluate Random Forest forecaster
│   ├── train_anomaly_model.py    # Workflow: Train & evaluate Isolation Forest detector
│   ├── create_topic.py         # Idempotent Kafka topic creator & inspector
│   ├── generate_demo_data.py   # Synthetic historical dataset generator
│   └── check_environment.py   # Diagnostic script verifying system readiness
├── data/
│   ├── raw/                    # Raw event append logs (CSV)
│   ├── processed/              # Spark streaming output (Parquet & CSV) & clustered CSV
│   └── demo/                   # Labelled synthetic demonstration datasets
├── models/                     # Saved ML models, encoders, and evaluation metadata
│   ├── congestion_model.joblib
│   ├── road_encoder.joblib
│   ├── model_metadata.json
│   ├── isolation_forest_model.joblib
│   ├── anomaly_metadata.json
│   └── kmeans_traffic_model.joblib
├── logs/                       # Application execution logs
└── tests/
    ├── __init__.py
    ├── test_anomaly_detection.py      # Tests for Isolation Forest feature scaling & outlier detection
    ├── test_congestion_prediction.py  # Tests for Random Forest target construction & chronological split
    ├── test_clustering.py             # Tests for K-Means preprocessing, scaling, dynamic labels & fallback
    ├── test_congestion.py             # Tests for thresholds, closures & zero-speed
    └── test_data_validation.py        # Tests for duplicates, formatting & tolerance
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

## 🔮 Supervised Machine Learning: Random Forest Congestion Forecaster

In addition to profiling current traffic with K-Means, the system provides proactive **15-minute future congestion forecasting** using a supervised `RandomForestClassifier` (`src/congestion_prediction.py`).

### 1. Prediction Target & Lookahead Construction
- **Target Variable**: Congestion category approximately 15 minutes into the future: `low`, `moderate`, `high`, or `closed`.
- **Lookahead Matching Without Leakage**: For each observation at time $t$ on road $R$, the training pipeline searches strictly for a subsequent observation on the **same road** at $t_{\text{future}}$ such that:
  $$10 \text{ minutes} \le (t_{\text{future}} - t) \le 20 \text{ minutes}$$
- **Zero Lookahead Leakage**: Features at time $t$ only contain data known at or before time $t$. Future speeds, future ratios, and future targets are never included in the input feature matrix.

### 2. Feature Engineering
The model uses 12 features derived from genuine domain measurements:
- **Spatial**: `road_encoded` (Label-encoded Pune road identifier).
- **Temporal**: `hour` (continuous hour), `sin_hour` & `cos_hour` (cyclical 24-hour trigonometric encoding), `day_of_week` (0–6), and `is_weekend` (binary flag).
- **Kinematic**: `current_speed`, `free_flow_speed`, `congestion_ratio` (safely guarded against division-by-zero).
- **Sequential Lags**: `prev_speed` (observed speed from immediately preceding time step on the same road).
- **Operational**: `confidence` (sensor confidence), `road_closure` (closure flag).

### 3. Chronological Validation & Baseline Comparison
- **Chronological Train/Test Split**: Standard random splits leak future patterns into past training folds. The pipeline splits the dataset strictly chronologically (first 80% for training, remaining 20% for testing).
- **Majority-Class Baseline**: The model is evaluated against a `DummyClassifier(strategy="most_frequent")` baseline to ensure genuine learning over naive guessing.
- **Class Balancing**: Uses `class_weight="balanced"` to prevent majority-class bias.
- **Metrics Tracked**: Test accuracy, macro F1-score, weighted F1-score, macro precision, recall, and a full confusion matrix.

### 4. Training Command
```powershell
python scripts/train_prediction_model.py --horizon 15 --tolerance 5
```

---

## 🚨 Unsupervised Machine Learning: Isolation Forest Anomaly Detection

Traffic networks frequently encounter unusual disturbances such as sudden bottlenecks, extreme speed drops, or malfunctioning sensors. The system incorporates **Isolation Forest** (`src/anomaly_detection.py`) for automated outlier detection.

### 1. Detection Principle
Isolation Forest isolates anomalies by randomly partitioning feature trees. Because anomalous observations have extreme or rare feature combinations (e.g., severe speed drops below baseline capacity), they require significantly fewer tree splittings to isolate compared to normal dense clusters.

### 2. Features Selected
- `current_speed`: Absolute travel speed in km/h.
- `free_flow_speed`: Baseline uncongested road limit.
- `congestion_ratio`: Relative speed ratio.
- `speed_deficit`: $\max(0, \text{free\_flow\_speed} - \text{current\_speed})$ (quantifies speed deficit below free flow).

### 3. Key Parameters & Outputs
- **Contamination Factor (`contamination=0.05`)**: Configured to flag approximately 5% of observations as statistical outliers.
- **Anomaly Score ($0.0 - 1.0$)**: Normalized continuous score where higher values represent greater deviation.
- **Severity Flagging**: Categorized into `Normal`, `Moderate Anomaly`, and `Severe Anomaly`.
- **Causal Disclaimer**:
  > ⚠️ **Important**: Isolation Forest identifies *statistical pattern anomalies*, NOT physical causes. An anomaly flags unusual traffic behavior, but cannot diagnose whether it was caused by an accident, road construction, weather, or a VIP convoy without external sensor inputs.

### 4. Training Command
```powershell
python scripts/train_anomaly_model.py --contamination 0.05
```

---

## 📊 Comparative Machine Learning Architecture

The system synthesizes three complementary machine-learning paradigms:

| Dimension | K-Means Clustering | Random Forest Classifier | Isolation Forest |
| :--- | :--- | :--- | :--- |
| **Learning Paradigm** | **Unsupervised Learning** | **Supervised Learning** | **Unsupervised Learning** |
| **Core Objective** | Group current road conditions into operational regimes | Forecast upcoming congestion ~15 minutes in advance | Detect rare, extreme traffic deviations & bottlenecks |
| **Input Features** | `current_speed`, `free_flow_speed`, `congestion_ratio` | Speeds, ratios, cyclical hours, road encoding, lag speed | Speeds, ratios, and speed deficit from baseline |
| **Target Variable** | None (Discovered mathematically via centroids) | Ground-truth congestion category at $t + 15\text{m}$ | None (Partition path length in isolation trees) |
| **Dynamic Interpretation**| Clusters dynamically mapped to Heavy, Moderate, Light | Predicts class: `low`, `moderate`, `high`, `closed` | Flagged as `Normal` or `Anomalous` with anomaly score |
| **Operational Role** | Real-time traffic state profiling | Proactive delay management | Incident & outlier flagging |

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
All **32 automated unit tests** verify:
- **Isolation Forest Anomaly Detection (`test_anomaly_detection.py`) [5 tests]**:
  - Feature extraction, division-by-zero guards, and speed deficit computation
  - Screen out of non-finite, negative, and zero free-flow speeds
  - Proper flagging of injected extreme bottlenecks as anomalies
  - Model serialization and deserialization via joblib
  - Defensive helper fallback on empty/corrupted data
- **Random Forest Congestion Prediction (`test_congestion_prediction.py`) [5 tests]**:
  - Feature matrix extraction, cyclical trigonometric encoding, and historical lag construction
  - Zero lookahead leakage in $(15 \pm 5 \text{ min})$ future target matching
  - Error handling when training dataset is undersized
  - Inference output schema, class probabilities, and prediction format
  - Safe prediction handling when baseline speed is zero or negative
- **K-Means Clustering (`test_clustering.py`) [10 tests]**:
  - Feature extraction, standardization via StandardScaler, and ratio computation
  - Missing, invalid, and non-finite value exclusion
  - Zero and negative free-flow speed protection
  - Insufficient observation threshold handling
  - Dynamic cluster interpretation (Heavy, Moderate, Light) based on learned centroid ratio and speed
  - Centroid and cluster statistics verification in physical units
  - Reproducibility with fixed random seed (`random_state=42`)
  - Model serialization and deserialization
  - Dashboard helper fallback on empty, corrupted, or missing data
- **Congestion Classification Rules (`test_congestion.py`) [6 tests]**:
  - Threshold boundaries (`low`, `moderate`, `high`)
  - Physical road closure override behavior
  - Missing, negative, and zero speed edge cases
  - Custom threshold dictionary support
- **Data Pipelines & Target Generation (`test_data_validation.py`) [6 tests]**:
  - Field validation and ISO 8601 UTC timestamp parsing
  - Malformed event rejection
  - Chronological sorting and duplicate elimination
  - Tolerance matching without lookahead leakage
  - Chronological train/test split integrity

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
