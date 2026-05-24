"""Plotly visualization helpers for PersonaX AI."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP


PLOT_TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = ["#19d3f3", "#ff4ecd", "#ffe66d", "#7cff6b", "#ff8c42", "#a78bfa", "#44d7b6"]


def reduce_dimensions(scaled_df: pd.DataFrame, method: str, random_state: int = 42) -> pd.DataFrame:
    X = scaled_df.values
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "t-SNE":
        perplexity = min(35, max(5, (len(scaled_df) - 1) // 3))
        reducer = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
    else:
        reducer = UMAP(n_components=2, n_neighbors=min(20, max(5, len(scaled_df) // 20)), min_dist=0.12, random_state=random_state)

    points = reducer.fit_transform(X)
    return pd.DataFrame({"x": points[:, 0], "y": points[:, 1]})


def cluster_scatter(plot_df: pd.DataFrame, method: str) -> go.Figure:
    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="persona",
        symbol="is_anomaly",
        hover_data={
            "user_id": True,
            "cluster_id": True,
            "persona": True,
            "anomaly_score": ":.1f",
            "x": False,
            "y": False,
            "is_anomaly": False,
        },
        color_discrete_sequence=COLOR_SEQUENCE,
        template=PLOT_TEMPLATE,
        title=f"{method} Behavioral Map",
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.8, color="rgba(255,255,255,0.55)")), selector=dict(mode="markers"))
    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="Persona",
    )
    return fig


def distribution_chart(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby(["cluster_id", "persona"]).size().reset_index(name="users")
    fig = px.bar(
        counts,
        x="persona",
        y="users",
        color="persona",
        color_discrete_sequence=COLOR_SEQUENCE,
        template=PLOT_TEMPLATE,
        title="Cluster Distribution",
    )
    fig.update_layout(showlegend=False, height=360, margin=dict(l=20, r=20, t=55, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def correlation_heatmap(df: pd.DataFrame, selected_features: list[str]) -> go.Figure:
    corr = df[selected_features].corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template=PLOT_TEMPLATE,
        title="Behavior Correlation Matrix",
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=55, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def cluster_profile_heatmap(summary: pd.DataFrame, selected_features: list[str]) -> go.Figure:
    profile = summary.set_index("persona")[selected_features]
    normalized = (profile - profile.mean()) / profile.std(ddof=0).replace(0, 1)
    fig = px.imshow(
        normalized,
        text_auto=".1f",
        color_continuous_scale="Tealrose",
        template=PLOT_TEMPLATE,
        title="Persona Behavior Intensity Heatmap",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig
