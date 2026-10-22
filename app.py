"""
Pune Traffic Intelligence Dashboard.
A Streamlit and Plotly interactive command center displaying real-time
and historical traffic flow metrics, congestion levels, map visualizations,
and predictive ML forecasting for Pune's road corridors.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    PUNE_ROAD_POINTS, PROCESSED_DATA_DIR, DEMO_DATA_DIR, RAW_DATA_DIR, MODELS_DIR,
    KMEANS_DEFAULT_CLUSTERS, ISOLATION_FOREST_CONTAMINATION
)
from src.data_utils import clean_and_prepare_dataset
from src.clustering import (
    TrafficKMeansClusterer,
    cluster_traffic_data,
    LABEL_HEAVY,
    LABEL_MODERATE,
    LABEL_LIGHT
)
from src.congestion_prediction import CongestionPredictor, TrafficPredictor
from src.anomaly_detection import TrafficAnomalyDetector, detect_traffic_anomalies

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="Pune Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark / Navy Modern CSS Theme
st.markdown("""
<style>
    /* Dark Navy Background */
    .stApp {
        background-color: #0b132b;
        color: #e0e6ed;
    }
    
    /* Header & Title Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #f0f4f8 !important;
        font-family: 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #1c2541;
        border: 1px solid #3a506b;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #48cae4 !important;
        font-weight: 700;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #adb5bd !important;
        font-weight: 500;
    }

    /* Badges */
    .badge-live {
        background-color: #1b4332;
        color: #74c69d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #40916c;
    }
    .badge-demo {
        background-color: #4a2810;
        color: #f4a261;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #e76f51;
    }
    .badge-spark {
        background-color: #14213d;
        color: #e5e5e5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #fca311;
    }
    .badge-anomaly {
        background-color: #4a151b;
        color: #ff6b6b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #e71d36;
    }
    .badge-normal {
        background-color: #0f3d3e;
        color: #74c69d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #2ec4b6;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Data Loader
# ==============================================================================
@st.cache_data(ttl=5)
def load_traffic_data():
    """
    Loads traffic records prioritizing live Spark outputs over demo records:
    1. Processed CSV from Spark streaming (data/processed/pune_traffic_latest_stream.csv)
    2. Processed Parquet directory (data/processed/parquet_stream)
    3. Demo dataset (data/demo/pune_traffic_historical_demo.csv)
    4. Raw events CSV (data/raw/traffic_raw_events.csv)
    """
    spark_csv = PROCESSED_DATA_DIR / "pune_traffic_latest_stream.csv"
    spark_parquet = PROCESSED_DATA_DIR / "parquet_stream"
    demo_csv = DEMO_DATA_DIR / "pune_traffic_historical_demo.csv"
    raw_csv = RAW_DATA_DIR / "traffic_raw_events.csv"

    source_desc = "No Data Found"
    data_mode = "unknown"
    df = pd.DataFrame()

    if spark_csv.exists() and spark_csv.stat().st_size > 100:
        try:
            df = pd.read_csv(spark_csv)
            source_desc = "Apache Spark Streaming Output (Live Stream)"
            data_mode = "spark_live"
        except Exception:
            pass

    if df.empty and spark_parquet.exists() and any(spark_parquet.glob("*.parquet")):
        try:
            df = pd.read_parquet(spark_parquet)
            source_desc = "Apache Spark Streaming Parquet Warehouse"
            data_mode = "spark_parquet"
        except Exception:
            pass

    if df.empty and demo_csv.exists():
        try:
            df = pd.read_csv(demo_csv)
            source_desc = "Synthetic Demo Dataset (Offline Evaluation)"
            data_mode = "demo"
        except Exception:
            pass

    if df.empty and raw_csv.exists() and raw_csv.stat().st_size > 100:
        try:
            df = pd.read_csv(raw_csv)
            source_desc = "Raw Producer CSV Log"
            data_mode = "raw"
        except Exception:
            pass

    if not df.empty:
        df = clean_and_prepare_dataset(df)

    return df, source_desc, data_mode


# ==============================================================================
# Color Map for Congestion Levels & K-Means Clusters
# ==============================================================================
COLOR_MAP = {
    "low": "#2ec4b6",       # Vibrant Teal Green
    "moderate": "#ff9f1c",  # Warm Amber Orange
    "high": "#e71d36",      # Vivid Crimson Red
    "closed": "#7b2cbf",    # Deep Purple
    "unknown": "#6c757d"    # Neutral Slate Gray
}

CLUSTER_COLOR_MAP = {
    LABEL_LIGHT: "#2ec4b6",     # Teal Green
    LABEL_MODERATE: "#ff9f1c",  # Amber Orange
    LABEL_HEAVY: "#e71d36",     # Vivid Red
    "Unknown": "#6c757d"
}


# ==============================================================================
# Main Dashboard UI
# ==============================================================================
def main():
    # Header Section
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("🚦 Pune Traffic Intelligence & Congestion Prediction")
        st.caption("IoT-Enabled Real-Time Big Data Analytics powered by Apache Kafka, Apache Spark, and Machine Learning")
    with col_t2:
        st.write("")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Load data
    df, source_desc, data_mode = load_traffic_data()

    # Data Source Banner
    if data_mode == "spark_live" or data_mode == "spark_parquet":
        badge_html = f'<span class="badge-spark">⚡ LIVE PIPELINE</span> <span style="margin-left: 10px; color: #adb5bd;">{source_desc}</span>'
    elif data_mode == "demo":
        badge_html = f'<span class="badge-demo">🧪 DEMO MODE</span> <span style="margin-left: 10px; color: #adb5bd;">{source_desc} (Run producer.py to stream live events)</span>'
    elif not df.empty:
        badge_html = f'<span class="badge-live">📡 LOCAL DATA</span> <span style="margin-left: 10px; color: #adb5bd;">{source_desc}</span>'
    else:
        badge_html = '<span style="color: #e71d36; font-weight: 600;">⚠️ No traffic data detected</span>'

    st.markdown(f"<div style='margin-bottom: 20px; padding: 10px; background-color: #141f36; border-radius: 8px;'>{badge_html}</div>", unsafe_allow_html=True)

    # Empty State Guidance
    if df.empty:
        st.warning("No traffic observations found in data/processed/ or data/demo/.")
        st.info(
            "### 🚀 How to start the system:\n"
            "1. **Generate demo dataset** (for immediate exploration):\n"
            "   ```powershell\n"
            "   python scripts/generate_demo_data.py\n"
            "   ```\n"
            "2. **Or start full Kafka & Spark pipeline**:\n"
            "   ```powershell\n"
            "   # Terminal 1: Start Kafka\n"
            "   docker compose up -d\n"
            "   # Terminal 2: Run Spark Streaming\n"
            "   python spark_streaming.py\n"
            "   # Terminal 3: Run Producer\n"
            "   python producer.py\n"
            "   ```"
        )
        return

    # Sidebar Filters
    st.sidebar.header("🔍 Filter Options")
    road_options = ["All Roads"] + sorted(list(df["road_id"].unique()))
    selected_road = st.sidebar.selectbox("Filter by Road Corridor:", road_options)

    st.sidebar.markdown("---")
    st.sidebar.header("🧩 K-Means Clustering")
    k_clusters = st.sidebar.slider(
        "Clusters (K):",
        min_value=2,
        max_value=5,
        value=KMEANS_DEFAULT_CLUSTERS,
        step=1,
        help="Number of clusters to group traffic observations into (default: 3)."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🚨 Anomaly Detection")
    iso_contamination = st.sidebar.slider(
        "Expected Contamination (%):",
        min_value=1.0,
        max_value=15.0,
        value=float(ISOLATION_FOREST_CONTAMINATION * 100),
        step=0.5,
        help="Expected percentage of unusual traffic anomalies (default: 5.0%)."
    ) / 100.0

    # Filter by Road
    if selected_road != "All Roads":
        filtered_df = df[df["road_id"] == selected_road].copy()
    else:
        filtered_df = df.copy()

    # ==============================================================================
    # 1. KPI Cards
    # ==============================================================================
    latest_snapshot = df.sort_values("timestamp").groupby("road_id").last().reset_index()
    
    total_obs = len(filtered_df)
    avg_speed = filtered_df["current_speed"].mean()
    high_cong_pct = (filtered_df["congestion_level"].isin(["high", "closed"]).mean()) * 100
    active_closures = int((latest_snapshot["road_closure"] == True).sum())

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Records", f"{total_obs:,}")
    with kpi2:
        st.metric("Avg Observed Speed", f"{avg_speed:.1f} km/h")
    with kpi3:
        st.metric("Congestion Index", f"{high_cong_pct:.1f}%", help="% observations with High Congestion or Closures")
    with kpi4:
        st.metric("Active Road Closures", f"{active_closures}", delta_color="inverse")

    st.markdown("---")

    # ==============================================================================
    # 2. Interactive Map & Congestion Distribution
    # ==============================================================================
    col_map, col_donut = st.columns([3, 2])

    with col_map:
        st.subheader("🗺️ Pune Live Traffic Map")
        map_df = latest_snapshot.copy()
        if selected_road != "All Roads":
            map_df = map_df[map_df["road_id"] == selected_road]

        # Generate OpenStreetMap scatter plot
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="congestion_level",
            size="free_flow_speed",
            color_discrete_map=COLOR_MAP,
            hover_name="road_name",
            hover_data={
                "current_speed": True,
                "free_flow_speed": True,
                "congestion_level": True,
                "congestion_ratio": ":.2f",
                "road_closure": True,
                "latitude": False,
                "longitude": False
            },
            zoom=11.2,
            center={"lat": 18.5204, "lon": 73.8567},
            mapbox_style="open-street-map",
            height=450
        )
        fig_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor="#0b132b",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color="#e0e6ed")
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_donut:
        st.subheader("📊 Congestion Breakdown")
        dist = filtered_df["congestion_level"].value_counts().reset_index()
        dist.columns = ["Level", "Count"]

        fig_donut = px.pie(
            dist,
            names="Level",
            values="Count",
            color="Level",
            color_discrete_map=COLOR_MAP,
            hole=0.55,
            height=450
        )
        fig_donut.update_layout(
            paper_bgcolor="#0b132b",
            font=dict(color="#e0e6ed"),
            legend=dict(orientation="v", font=dict(color="#e0e6ed")),
            margin={"r": 20, "t": 20, "l": 20, "b": 20}
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("---")

    # ==============================================================================
    # 3. Time Series & Hourly Traffic Pattern Analysis
    # ==============================================================================
    col_ts, col_hour = st.columns([3, 2])

    with col_ts:
        st.subheader("📈 Speed Fluctuations Over Time")
        ts_sample = filtered_df.sort_values("timestamp").tail(200)

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=ts_sample["timestamp"],
            y=ts_sample["current_speed"],
            mode="lines+markers",
            name="Current Speed",
            line=dict(color="#48cae4", width=2.5),
            marker=dict(size=4)
        ))
        fig_ts.add_trace(go.Scatter(
            x=ts_sample["timestamp"],
            y=ts_sample["free_flow_speed"],
            mode="lines",
            name="Free Flow Speed",
            line=dict(color="#adb5bd", width=1.5, dash="dash")
        ))
        fig_ts.update_layout(
            paper_bgcolor="#0b132b",
            plot_bgcolor="#162038",
            font=dict(color="#e0e6ed"),
            xaxis=dict(gridcolor="#28385e", title="Timestamp (UTC)"),
            yaxis=dict(gridcolor="#28385e", title="Speed (km/h)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=30, b=30),
            height=380
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_hour:
        st.subheader("🕒 Hour-of-Day Congestion Profile")
        filtered_df["hour"] = pd.to_datetime(filtered_df["timestamp"]).dt.hour
        hourly = filtered_df.groupby("hour")["current_speed"].mean().reset_index()

        fig_hour = px.bar(
            hourly,
            x="hour",
            y="current_speed",
            color="current_speed",
            color_continuous_scale="Tealrose",
            labels={"hour": "Hour of Day (UTC)", "current_speed": "Avg Speed (km/h)"},
            height=380
        )
        fig_hour.update_layout(
            paper_bgcolor="#0b132b",
            plot_bgcolor="#162038",
            font=dict(color="#e0e6ed"),
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#28385e", dtick=2),
            yaxis=dict(gridcolor="#28385e"),
            margin=dict(l=40, r=20, t=30, b=30)
        )
        st.plotly_chart(fig_hour, use_container_width=True)

    st.markdown("---")

    # ==============================================================================
    # 4. Traffic Clustering — K-Means (Unsupervised Data Mining)
    # ==============================================================================
    st.subheader("🧩 Traffic Clustering — K-Means")
    st.write(
        "Discovers latent traffic patterns across Pune road corridors by grouping multidimensional "
        "observations (`current_speed`, `free_flow_speed`, and `congestion_ratio`) into unsupervised clusters. "
        "Unlike supervised forecasting models, K-Means does not predict future congestion; it groups current "
        "or historical observations without relying on predefined target labels."
    )

    # Dataset scope selection & K parameter
    c_scope_col1, c_scope_col2 = st.columns([3, 1])
    with c_scope_col1:
        cluster_scope = st.radio(
            "Select Observations for Clustering:",
            [
                "Latest Snapshot (Latest observation per Pune road corridor)",
                "Full Active Stream (All current streaming records)"
            ],
            horizontal=True
        )
    with c_scope_col2:
        k_val = st.number_input(
            "Active Cluster Count (K):",
            min_value=2,
            max_value=5,
            value=k_clusters,
            step=1,
            help="Configurable K clusters (default: 3)."
        )

    # Determine candidate dataset for clustering
    if cluster_scope.startswith("Latest"):
        candidate_df = latest_snapshot.copy()
    else:
        candidate_df = filtered_df.copy()

    # Run clustering defensively
    annotated_clustered, cluster_stats, clusterer, cluster_err = cluster_traffic_data(
        df=candidate_df,
        n_clusters=int(k_val)
    )

    if cluster_err:
        st.warning(f"⚠️ Clustering unavailable: {cluster_err}")
        st.info(
            f"💡 **Tip**: Ensure at least {k_val} valid observations exist. "
            "Run `python producer.py` to stream live traffic data or `python scripts/generate_demo_data.py`."
        )
    else:
        source_badge = clusterer.data_source_label

        # 1. Summary Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Clusters Configured (K)", f"{k_val}")
        with m2:
            st.metric("Valid Observations Analyzed", f"{len(annotated_clustered):,}")
        with m3:
            st.metric("Analyzed Data Source", source_badge)

        # Observations assigned to each cluster
        st.markdown("##### Observations per Cluster:")
        cluster_cols = st.columns(len(cluster_stats))
        for idx, row in cluster_stats.iterrows():
            with cluster_cols[idx]:
                c_lbl = row["cluster_label"]
                c_count = int(row["count"])
                c_pct = float(row["percentage"])
                color_hex = CLUSTER_COLOR_MAP.get(c_lbl, "#48cae4")
                st.markdown(
                    f"<div style='background-color: #1c2541; border-left: 5px solid {color_hex}; "
                    f"padding: 10px 14px; border-radius: 6px;'>"
                    f"<div style='font-size: 0.85rem; color: #adb5bd;'>Cluster {row['cluster_id']}: <b>{c_lbl}</b></div>"
                    f"<div style='font-size: 1.5rem; font-weight: bold; color: {color_hex};'>{c_count} <span style='font-size: 0.9rem; color: #ced4da;'>({c_pct}%)</span></div>"
                    f"<div style='font-size: 0.8rem; color: #8ecae6;'>Mean Speed: {row['mean_current_speed']} km/h | Ratio: {row['mean_congestion_ratio']:.2f}</div>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        st.write("")

        # 2. Plotly Scatter Plot & Centroids
        col_c_chart, col_c_table = st.columns([3, 2])

        with col_c_chart:
            st.markdown("##### 📍 Multidimensional Cluster Scatter Plot")

            fig_cluster = px.scatter(
                annotated_clustered,
                x="current_speed",
                y="congestion_ratio",
                color="cluster_label",
                color_discrete_map=CLUSTER_COLOR_MAP,
                size="free_flow_speed",
                hover_name="road_name",
                hover_data={
                    "current_speed": True,
                    "free_flow_speed": True,
                    "congestion_ratio": ":.3f",
                    "cluster_id": True,
                    "cluster_label": True,
                    "data_source": True,
                },
                labels={
                    "current_speed": "Current Speed (km/h)",
                    "congestion_ratio": "Congestion Ratio (Current / Free-Flow)",
                    "cluster_label": "Interpreted Cluster"
                },
                height=420
            )

            centroids_df = clusterer.get_centroids()
            if not centroids_df.empty:
                fig_cluster.add_trace(go.Scatter(
                    x=centroids_df["current_speed"],
                    y=centroids_df["congestion_ratio"],
                    mode="markers+text",
                    marker=dict(symbol="star", size=16, color="#ffffff", line=dict(color="#000000", width=1.5)),
                    name="Cluster Centroids",
                    text=[f"Centroid {cid}" for cid in centroids_df["cluster_id"]],
                    textposition="top center",
                    hovertext=[
                        f"<b>Centroid - {lbl}</b><br>Speed: {spd} km/h<br>Free-Flow: {ff} km/h<br>Ratio: {r:.4f}"
                        for lbl, spd, ff, r in zip(
                            centroids_df["cluster_label"],
                            centroids_df["current_speed"],
                            centroids_df["free_flow_speed"],
                            centroids_df["congestion_ratio"]
                        )
                    ]
                ))

            fig_cluster.update_layout(
                paper_bgcolor="#0b132b",
                plot_bgcolor="#162038",
                font=dict(color="#e0e6ed"),
                xaxis=dict(gridcolor="#28385e", title="Current Speed (km/h)"),
                yaxis=dict(gridcolor="#28385e", title="Congestion Ratio (Current / Free-Flow)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e0e6ed")),
                margin=dict(l=30, r=20, t=30, b=30)
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            st.caption(
                "ℹ️ **Feature Space Explanation**: K-Means clustered the data using 3 normalized numerical features: "
                "`current_speed`, `free_flow_speed`, and `congestion_ratio` (scaled via `StandardScaler`). "
                "The 2D projection above displays Current Speed vs. Congestion Ratio, with white star markers "
                "representing learned multidimensional cluster centroids."
            )

        with col_c_table:
            st.markdown("##### 📐 Learned Cluster Centroid Statistics")
            display_stats = cluster_stats.rename(columns={
                "cluster_id": "Cluster ID",
                "cluster_label": "Interpreted Label",
                "count": "Count",
                "percentage": "Share (%)",
                "mean_current_speed": "Avg Speed (km/h)",
                "mean_free_flow_speed": "Avg Baseline (km/h)",
                "mean_congestion_ratio": "Avg Ratio"
            })
            st.dataframe(display_stats, use_container_width=True, hide_index=True)

            st.markdown("""
            **🔍 How to Interpret the Results:**
            - **Heavy Congestion**: Characterized by the lowest congestion ratio and slowest speeds. Traffic is significantly restricted below baseline capacity.
            - **Moderate Congestion**: Characterized by intermediate speed ratios. Vehicles experience noticeable delays but are still moving.
            - **Light Congestion**: Characterized by high congestion ratios (close to 1.0) and optimal travel speeds near or at free-flow limits.
            - **Dynamic Labeling**: Cluster IDs (0, 1, 2) do not inherently mean heavy or light; labels are assigned automatically by sorting learned centroid throughput ratios.
            """)

        # 3. Table of Clustered Observations
        st.markdown("##### 🛣️ Road-Level Clustered Observations")
        road_cluster_cols = [
            "road_name", "current_speed", "free_flow_speed",
            "congestion_ratio", "cluster_id", "cluster_label", "data_source", "timestamp"
        ]
        avail_rc_cols = [c for c in road_cluster_cols if c in annotated_clustered.columns]
        sorted_clustered_table = annotated_clustered.sort_values(
            by=["congestion_ratio", "current_speed"], ascending=[True, True]
        )[avail_rc_cols]

        st.dataframe(sorted_clustered_table, use_container_width=True, height=260)

    st.markdown("---")

    # ==============================================================================
    # 5. Congestion Prediction — Random Forest (Supervised Forecasting)
    # ==============================================================================
    st.subheader("🔮 Congestion Prediction — Random Forest")
    st.write(
        "Forecasts anticipated congestion **~15 minutes into the future** "
        "using a supervised `RandomForestClassifier` trained with chronological validation, "
        "cyclical hour encoding, road embeddings, and historical speed lags."
    )

    predictor = TrafficPredictor()
    if not predictor.is_ready():
        st.warning("⚠️ Trained prediction model not found in `models/`. Run `python scripts/train_prediction_model.py` to train and enable predictions.")
    else:
        meta = predictor.metadata
        model_badge = "🧪 Synthetic Demo Dataset" if meta.get("is_synthetic") else "📡 Live TomTom API Dataset"
        
        # Display Training & Evaluation Metrics
        acc = meta.get("test_accuracy", 0) * 100
        base_acc = meta.get("baseline_accuracy", 0) * 100
        macro_f1 = meta.get("macro_f1", 0)
        weighted_f1 = meta.get("weighted_f1", 0)

        rf_m1, rf_m2, rf_m3, rf_m4 = st.columns(4)
        with rf_m1:
            st.metric("Test Accuracy", f"{acc:.1f}%", help="Chronologically evaluated test accuracy on held-out observations.")
        with rf_m2:
            st.metric("Majority Baseline", f"{base_acc:.1f}%", help="Accuracy of a naive DummyClassifier always predicting the majority class.")
        with rf_m3:
            st.metric("Macro F1-Score", f"{macro_f1:.3f}", help="Unweighted mean of F1-scores across all classes.")
        with rf_m4:
            st.metric("Evaluation Dataset", model_badge)

        st.caption(
            f"ℹ️ **Evaluation Integrity**: Model was evaluated on **{meta.get('test_samples', 0):,} held-out observations** "
            f"using strict chronological splitting (up to split time: `{meta.get('split_timestamp', 'N/A')}`) "
            f"to guarantee zero future lookahead leakage."
        )

        # Interactive Forecast Simulator
        st.markdown("##### 🧪 15-Minute Future Horizon Simulator")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            pred_road = st.selectbox("Target Corridor:", [pt["road_id"] for pt in PUNE_ROAD_POINTS], index=0)
        with p_col2:
            default_free = next((pt["typical_free_flow_speed"] for pt in PUNE_ROAD_POINTS if pt["road_id"] == pred_road), 50.0)
            in_cur_spd = st.slider("Observed Current Speed (km/h):", min_value=0.0, max_value=80.0, value=22.0, step=1.0)
        with p_col3:
            in_free_spd = st.slider("Baseline Free-Flow Speed (km/h):", min_value=20.0, max_value=80.0, value=default_free, step=1.0)

        sim_button = st.button("🔮 Compute 15-Minute Forecast", type="primary")
        if sim_button:
            pred_res = predictor.predict(
                road_id=pred_road,
                current_speed=in_cur_spd,
                free_flow_speed=in_free_spd
            )

            level_pred = pred_res.get("predicted_congestion_level_15m", "unknown").lower()
            prob_dict = pred_res.get("probabilities", {})
            forecast_time = pred_res.get("forecast_timestamp_utc", "t + 15m")

            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                st.markdown(f"**Predicted Status (at {forecast_time[:19]} UTC):**")
                color = COLOR_MAP.get(level_pred, "#6c757d")
                st.markdown(
                    f"<div style='background-color: {color}; color: white; padding: 18px 20px; "
                    f"border-radius: 8px; font-size: 1.4rem; font-weight: bold; text-align: center;'>"
                    f"{level_pred.upper()} CONGESTION</div>",
                    unsafe_allow_html=True
                )
                curr_ratio = pred_res.get('current_ratio', 0.0)
                st.write(f"Current Speed Ratio: **{curr_ratio:.2f}** | Forecast Horizon: **~15 mins**")

            with col_res2:
                st.markdown("**Class Confidence Distribution:**")
                prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Level", "Probability"])
                fig_prob = px.bar(
                    prob_df,
                    x="Probability",
                    y="Level",
                    orientation="h",
                    color="Level",
                    color_discrete_map=COLOR_MAP,
                    text="Probability",
                    height=200
                )
                fig_prob.update_layout(
                    paper_bgcolor="#0b132b",
                    plot_bgcolor="#162038",
                    font=dict(color="#e0e6ed"),
                    xaxis=dict(range=[0, 1], gridcolor="#28385e"),
                    yaxis=dict(gridcolor="#28385e"),
                    margin=dict(l=10, r=20, t=10, b=10)
                )
                st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # ==============================================================================
    # 6. Traffic Anomaly Detection — Isolation Forest
    # ==============================================================================
    st.subheader("🚨 Traffic Anomaly Detection — Isolation Forest")
    st.write(
        "Detects rare and unexpected traffic observations (such as sudden speed drops, severe bottlenecks "
        "relative to baseline capacity, or unusual sensor readings) using an unsupervised `IsolationForest`."
    )

    # Anomaly scope selection
    anom_annotated, anom_stats, anom_detector, anom_err = detect_traffic_anomalies(
        df=candidate_df,
        contamination=float(iso_contamination),
        use_saved_model=True
    )

    if anom_err:
        st.warning(f"⚠️ Anomaly detection unavailable: {anom_err}")
    else:
        # Anomaly KPI Cards
        ak1, ak2, ak3, ak4 = st.columns(4)
        with ak1:
            st.metric("Analyzed Records", f"{anom_stats.get('total_observations', 0):,}")
        with ak2:
            st.metric("Normal Observations", f"{anom_stats.get('normal_count', 0):,}")
        with ak3:
            st.metric("Flagged Anomalies", f"{anom_stats.get('anomaly_count', 0):,}", delta_color="inverse")
        with ak4:
            st.metric("Anomaly Rate", f"{anom_stats.get('anomaly_percentage', 0):.2f}%", help="Percentage of observations flagged as statistical anomalies.")

        # Interactive Anomaly Plotly Chart & Table
        col_anom_chart, col_anom_exp = st.columns([3, 2])

        with col_anom_chart:
            st.markdown("##### 📍 Statistical Outlier Visualization")
            
            anom_color_map = {
                "Normal": "#2ec4b6",
                "Anomalous": "#e71d36"
            }

            fig_anom = px.scatter(
                anom_annotated,
                x="current_speed",
                y="congestion_ratio",
                color="anomaly_label",
                symbol="anomaly_label",
                symbol_map={"Normal": "circle", "Anomalous": "diamond"},
                color_discrete_map=anom_color_map,
                size="speed_deficit",
                size_max=16,
                hover_name="road_name",
                hover_data={
                    "current_speed": True,
                    "free_flow_speed": True,
                    "congestion_ratio": ":.3f",
                    "speed_deficit": ":.1f",
                    "anomaly_score": ":.3f",
                    "anomaly_severity": True,
                    "data_source": True
                },
                labels={
                    "current_speed": "Current Speed (km/h)",
                    "congestion_ratio": "Congestion Ratio",
                    "anomaly_label": "Observation Status"
                },
                height=400
            )
            fig_anom.update_layout(
                paper_bgcolor="#0b132b",
                plot_bgcolor="#162038",
                font=dict(color="#e0e6ed"),
                xaxis=dict(gridcolor="#28385e", title="Current Speed (km/h)"),
                yaxis=dict(gridcolor="#28385e", title="Congestion Ratio (Current / Free-Flow)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#e0e6ed")),
                margin=dict(l=30, r=20, t=30, b=30)
            )
            st.plotly_chart(fig_anom, use_container_width=True)

        with col_anom_exp:
            st.markdown("##### 🔍 How Isolation Forest Works")
            st.markdown("""
            - **Isolation Principle**: Normal traffic data points cluster densely around expected free-flow speeds. Anomalous observations (severe bottlenecks or sudden drops) are sparse and are isolated with very few random tree partitions.
            - **Anomaly Score ($0.0 - 1.0$)**: Scores $> 0.50$ indicate increasing statistical deviation from the baseline model.
            - **Causal Disclaimer**:
              > ⚠️ **Important**: Isolation Forest identifies *statistical pattern deviations*, NOT root causes. A flagged anomaly could represent a bottleneck, road closure, sensor calibration artifact, or VIP convoy, but should not automatically be labeled an accident without external verification.
            """)

        # Table of Flagged Anomalies
        st.markdown("##### 🚨 Flagged Anomalous Traffic Observations")
        flagged_df = anom_annotated[anom_annotated["is_anomaly"] == True]
        if flagged_df.empty:
            st.success("✅ No statistical anomalies detected in the current observation window.")
        else:
            anom_cols = [
                "timestamp", "road_name", "current_speed", "free_flow_speed",
                "congestion_ratio", "speed_deficit", "anomaly_score", "anomaly_severity", "data_source"
            ]
            avail_anom_cols = [c for c in anom_cols if c in flagged_df.columns]
            st.dataframe(
                flagged_df.sort_values("anomaly_score", ascending=False)[avail_anom_cols],
                use_container_width=True,
                height=220
            )

    st.markdown("---")

    # ==============================================================================
    # 7. Comparative Model Architecture & Evaluation
    # ==============================================================================
    st.subheader("📊 Comparative Machine Learning Architecture")
    st.write(
        "A multi-model synthesis combining unsupervised data mining, supervised forecasting, "
        "and unsupervised anomaly detection across Pune's road corridors."
    )

    comparison_data = [
        {
            "Algorithm": "K-Means Clustering",
            "Learning Paradigm": "Unsupervised Learning",
            "Core Objective": "Discover latent road condition regimes (Heavy, Moderate, Light)",
            "Key Features Used": "current_speed, free_flow_speed, congestion_ratio",
            "Output": "Discrete Cluster ID & Semantic Condition Label",
            "Operational Role": "Current Traffic State Profiling"
        },
        {
            "Algorithm": "Random Forest Classifier",
            "Learning Paradigm": "Supervised Learning",
            "Core Objective": "Predict future congestion level ~15 minutes in advance",
            "Key Features Used": "Speeds, ratios, cyclical hour (sin/cos), day of week, lag speed",
            "Output": "Future Congestion Category & Class Confidence",
            "Operational Role": "Proactive Congestion Forecasting"
        },
        {
            "Algorithm": "Isolation Forest",
            "Learning Paradigm": "Unsupervised Learning",
            "Core Objective": "Detect statistically rare speed drops and abnormal patterns",
            "Key Features Used": "current_speed, free_flow_speed, congestion_ratio, speed_deficit",
            "Output": "Anomaly Indicator (Normal/Anomalous) & Anomaly Score",
            "Operational Role": "Outlier & Incident Detection"
        }
    ]
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ==============================================================================
    # 8. Recent Observations Table
    # ==============================================================================
    st.subheader("📋 Latest Traffic Observations")
    display_cols = [
        "timestamp", "road_name", "current_speed", "free_flow_speed",
        "congestion_ratio", "congestion_level", "confidence", "road_closure", "data_source"
    ]
    avail_cols = [c for c in display_cols if c in filtered_df.columns]
    recent_table = filtered_df.sort_values("timestamp", ascending=False).head(25)[avail_cols]

    st.dataframe(recent_table, use_container_width=True, height=280)


if __name__ == "__main__":
    main()
