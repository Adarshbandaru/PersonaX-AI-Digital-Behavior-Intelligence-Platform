"""PersonaX AI - Digital Behavior Intelligence Platform."""

from __future__ import annotations

import streamlit as st
import pandas as pd

from anomaly_detection import anomaly_score_cards, detect_anomalies
from clustering import build_persona_summary, map_cluster_personas, run_clustering
from preprocessing import DEFAULT_FEATURES, available_numeric_features, generate_synthetic_behavior_data, preprocess_dataset
from utils import dataframe_to_csv_bytes, format_metric, generate_ai_insights
from visualization import cluster_profile_heatmap, cluster_scatter, correlation_heatmap, distribution_chart, reduce_dimensions


st.set_page_config(
    page_title="PersonaX AI",
    page_icon="PX",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
    --px-bg: #080b12;
    --px-panel: rgba(18, 24, 38, 0.82);
    --px-border: rgba(255, 255, 255, 0.10);
    --px-text: #eef4ff;
    --px-muted: #98a6bd;
    --px-accent: #19d3f3;
    --px-pink: #ff4ecd;
}
.stApp {
    background:
        radial-gradient(circle at 18% 12%, rgba(25, 211, 243, 0.18), transparent 28%),
        radial-gradient(circle at 90% 6%, rgba(255, 78, 205, 0.14), transparent 24%),
        linear-gradient(135deg, #080b12 0%, #101522 45%, #090d15 100%);
    color: var(--px-text);
}
[data-testid="stSidebar"] {
    background: rgba(8, 11, 18, 0.95);
    border-right: 1px solid var(--px-border);
}
.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}
.px-hero {
    padding: 1.35rem 1.5rem;
    border: 1px solid var(--px-border);
    background: linear-gradient(135deg, rgba(25, 211, 243, 0.14), rgba(255, 78, 205, 0.08));
    border-radius: 8px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.28);
}
.px-hero h1 {
    font-size: clamp(2rem, 5vw, 4rem);
    line-height: 1;
    margin: 0 0 .35rem 0;
    letter-spacing: 0;
}
.px-hero p {
    color: var(--px-muted);
    font-size: 1rem;
    margin: 0;
}
.metric-card {
    position: relative;
    overflow: hidden;
    min-height: 122px;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid var(--px-border);
    background: var(--px-panel);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
    animation: floatIn .55s ease both;
}
.metric-card::after {
    content: "";
    position: absolute;
    inset: auto -30% -50% 20%;
    height: 90px;
    background: linear-gradient(90deg, rgba(25, 211, 243, .20), rgba(255, 78, 205, .18));
    filter: blur(24px);
}
.metric-label {
    color: var(--px-muted);
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: .08em;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    margin-top: .25rem;
}
.metric-note {
    color: var(--px-muted);
    font-size: .86rem;
    margin-top: .35rem;
}
.insight {
    padding: .9rem 1rem;
    border: 1px solid var(--px-border);
    border-left: 3px solid var(--px-accent);
    background: rgba(18, 24, 38, 0.72);
    border-radius: 8px;
    margin-bottom: .7rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: .4rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: .65rem 1rem;
    background: rgba(255,255,255,0.04);
}
@keyframes floatIn {
    from { transform: translateY(10px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
</style>
"""


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def read_uploaded_csv(uploaded_file):
    if uploaded_file is None:
        return generate_synthetic_behavior_data(), "Synthetic demo dataset"
    return pd.read_csv(uploaded_file), uploaded_file.name


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

with st.sidebar:
    st.title("PersonaX AI")
    st.caption("Digital Behavior Intelligence Platform")
    uploaded_file = st.file_uploader("Upload behavior CSV", type=["csv"])

    raw_df, dataset_name = read_uploaded_csv(uploaded_file)
    numeric_options = available_numeric_features(raw_df)
    default_selection = [col for col in DEFAULT_FEATURES if col in numeric_options] or numeric_options[: min(8, len(numeric_options))]

    selected_features = st.multiselect("Behavior features", numeric_options, default=default_selection)
    algorithm = st.selectbox("Clustering algorithm", ["K-Means", "DBSCAN", "Agglomerative Clustering"])

    if algorithm in {"K-Means", "Agglomerative Clustering"}:
        n_clusters = st.slider("Number of personas", 2, 9, 5)
        eps = 1.25
        min_samples = 8
    else:
        n_clusters = 5
        eps = st.slider("DBSCAN epsilon", 0.30, 3.00, 1.25, 0.05)
        min_samples = st.slider("Minimum samples", 3, 25, 8)

    reduction_method = st.selectbox("2D projection", ["UMAP", "PCA", "t-SNE"])
    anomaly_method = st.selectbox("Anomaly model", ["Isolation Forest", "Local Outlier Factor"])
    contamination = st.slider("Expected abnormal users", 1, 18, 6) / 100

st.markdown(
    """
    <div class="px-hero">
        <h1>PersonaX AI</h1>
        <p>Discover behavioral personas, productivity patterns, and abnormal digital habits from raw activity data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not selected_features:
    st.error("Select at least one numeric feature in the sidebar.")
    st.stop()

with st.spinner("Analyzing digital behavior patterns..."):
    processed = preprocess_dataset(raw_df, selected_features)
    labels, model, metrics = run_clustering(
        processed.scaled_df,
        algorithm,
        n_clusters=n_clusters,
        eps=eps,
        min_samples=min_samples,
    )
    anomaly_labels, anomaly_scores, anomaly_model = detect_anomalies(
        processed.scaled_df,
        method=anomaly_method,
        contamination=contamination,
    )

    persona_summary = build_persona_summary(processed.clean_df, labels, selected_features)
    persona_map = map_cluster_personas(persona_summary)
    final_df = processed.clean_df.copy()
    final_df["cluster_id"] = labels
    final_df["persona"] = final_df["cluster_id"].map(persona_map).fillna("Unclassified Outliers")
    final_df["anomaly_score"] = anomaly_scores
    final_df["is_anomaly"] = anomaly_labels == -1

    embedded = reduce_dimensions(processed.scaled_df, reduction_method)
    plot_df = final_df[["user_id", "cluster_id", "persona", "anomaly_score", "is_anomaly"]].join(embedded)
    cards = anomaly_score_cards(final_df, anomaly_scores, anomaly_labels)
    insights = generate_ai_insights(persona_summary, cards)

st.write("")
metric_cols = st.columns(5)
with metric_cols[0]:
    metric_card("Dataset", f"{len(final_df):,}", dataset_name)
with metric_cols[1]:
    metric_card("Personas", format_metric(metrics["clusters"]), f"{algorithm}")
with metric_cols[2]:
    metric_card("Silhouette", format_metric(metrics["silhouette"]), "higher is cleaner")
with metric_cols[3]:
    metric_card("Davies-Bouldin", format_metric(metrics["davies_bouldin"]), "lower is better")
with metric_cols[4]:
    metric_card("Abnormal Users", f"{cards['abnormal_users']}", f"{cards['abnormal_rate']:.1f}% flagged")

tabs = st.tabs(["Behavior Map", "Personas", "Anomalies", "Data Lab", "Insights"])

with tabs[0]:
    left, right = st.columns([1.55, 1])
    with left:
        st.plotly_chart(cluster_scatter(plot_df, reduction_method), width="stretch")
    with right:
        st.plotly_chart(distribution_chart(final_df), width="stretch")
        if metrics["inertia"] is not None:
            metric_card("K-Means Inertia", format_metric(metrics["inertia"]), "compactness of clusters")
        else:
            metric_card("Noise Points", format_metric(metrics["noise_points"]), "DBSCAN outliers or none")

with tabs[1]:
    st.subheader("Automatic Persona Discovery")
    display_cols = ["cluster_id", "persona", "users", "dominant_behaviors"]
    numeric_display = [col for col in ["screen_time", "sleep_hours", "study_time", "productivity_score", "social_media_time", "app_switches"] if col in persona_summary]
    st.dataframe(
        persona_summary[display_cols + numeric_display].style.format({col: "{:.2f}" for col in numeric_display}),
        width="stretch",
        hide_index=True,
    )
    st.plotly_chart(cluster_profile_heatmap(persona_summary, selected_features), width="stretch")

with tabs[2]:
    anomaly_cols = st.columns(4)
    with anomaly_cols[0]:
        metric_card("Model", anomaly_method, "unsupervised detection")
    with anomaly_cols[1]:
        metric_card("Highest Score", f"{cards['highest_score']:.1f}", "0-100 normalized")
    with anomaly_cols[2]:
        metric_card("Highest Risk User", str(cards["highest_risk_user"]), "review recommended")
    with anomaly_cols[3]:
        metric_card("Flagged Share", f"{cards['abnormal_rate']:.1f}%", "dataset coverage")
    st.dataframe(
        final_df.sort_values("anomaly_score", ascending=False).head(25),
        width="stretch",
        hide_index=True,
    )

with tabs[3]:
    st.subheader("Dataset Preview")
    st.dataframe(final_df.head(60), width="stretch", hide_index=True)
    st.download_button(
        "Download processed CSV",
        data=dataframe_to_csv_bytes(final_df),
        file_name="personax_processed_behavior_data.csv",
        mime="text/csv",
    )
    heat_left, heat_right = st.columns(2)
    with heat_left:
        st.plotly_chart(correlation_heatmap(final_df, selected_features), width="stretch")
    with heat_right:
        st.write("Selected Features")
        st.dataframe(final_df[selected_features].describe().T, width="stretch")

with tabs[4]:
    st.subheader("AI Insights Panel")
    for insight in insights:
        st.markdown(f"<div class='insight'>{insight}</div>", unsafe_allow_html=True)
    st.info("These insights are generated from cluster means, anomaly scores, and behavior-risk heuristics for demo-ready interpretability.")
