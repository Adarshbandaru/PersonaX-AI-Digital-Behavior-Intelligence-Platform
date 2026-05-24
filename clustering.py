"""Clustering and persona discovery logic for PersonaX AI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


def run_clustering(
    scaled_df: pd.DataFrame,
    algorithm: str,
    n_clusters: int = 5,
    eps: float = 1.25,
    min_samples: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, object, dict[str, float | None]]:
    """Fit the selected clustering algorithm and return labels plus metrics."""
    X = scaled_df.values
    if algorithm == "K-Means":
        model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)
        inertia = float(model.inertia_)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        inertia = None
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        inertia = None

    metrics = compute_cluster_metrics(X, labels, inertia)
    return labels, model, metrics


def compute_cluster_metrics(X: np.ndarray, labels: np.ndarray, inertia: float | None) -> dict[str, float | None]:
    valid_mask = labels != -1
    unique_valid = np.unique(labels[valid_mask])
    if len(unique_valid) >= 2 and valid_mask.sum() > len(unique_valid):
        silhouette = float(silhouette_score(X[valid_mask], labels[valid_mask]))
        db_index = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
    else:
        silhouette = None
        db_index = None

    return {
        "silhouette": silhouette,
        "davies_bouldin": db_index,
        "inertia": inertia,
        "clusters": int(len(unique_valid)),
        "noise_points": int((labels == -1).sum()),
    }


def _persona_for(row: pd.Series) -> str:
    if row.name == -1:
        return "Unclassified Outliers"
    if row.get("study_time", 0) >= 4.5 and row.get("productivity_score", 0) >= 72:
        return "Deep Focus Users"
    if row.get("screen_time", 0) >= 7.5 and row.get("sleep_hours", 8) <= 6.2 and row.get("social_media_time", 0) >= 120:
        return "Late-Night Scrollers"
    if row.get("app_switches", 0) >= 145 or row.get("unlock_frequency", 0) >= 130:
        return "High Dopamine Switchers"
    if row.get("productivity_score", 0) >= 78:
        return "Productivity Driven Users"
    if row.get("gaming_time", 0) >= 120:
        return "Immersive Gamers"
    if row.get("sleep_hours", 0) >= 7 and row.get("screen_time", 99) <= 6 and row.get("productivity_score", 0) >= 62:
        return "Balanced Users"
    return "Mixed Behavior Users"


def dominant_behaviors(row: pd.Series) -> str:
    behaviors = []
    if row.get("screen_time", 0) >= 7:
        behaviors.append("heavy screen exposure")
    if row.get("unlock_frequency", 0) >= 120:
        behaviors.append("frequent phone checking")
    if row.get("app_switches", 0) >= 145:
        behaviors.append("rapid app switching")
    if row.get("social_media_time", 0) >= 120:
        behaviors.append("social media intensity")
    if row.get("gaming_time", 0) >= 110:
        behaviors.append("long gaming sessions")
    if row.get("study_time", 0) >= 4:
        behaviors.append("strong study focus")
    if row.get("sleep_hours", 0) < 6:
        behaviors.append("sleep compression")
    if row.get("productivity_score", 0) >= 75:
        behaviors.append("high productivity")
    return ", ".join(behaviors[:4]) if behaviors else "moderate, balanced usage"


def build_persona_summary(df: pd.DataFrame, labels: np.ndarray, selected_features: list[str]) -> pd.DataFrame:
    """Create human-readable persona summaries from cluster-level means."""
    working = df.copy()
    working["cluster_id"] = labels
    numeric_features = [feature for feature in selected_features if feature in working.columns]
    grouped = working.groupby("cluster_id")[numeric_features].mean(numeric_only=True)
    counts = working.groupby("cluster_id").size().rename("users")
    summary = grouped.join(counts)
    summary["persona"] = summary.apply(_persona_for, axis=1)
    summary["dominant_behaviors"] = summary.apply(dominant_behaviors, axis=1)
    summary = summary.reset_index()
    columns = ["cluster_id", "persona", "users", "dominant_behaviors"] + numeric_features
    return summary[columns].sort_values("cluster_id")


def map_cluster_personas(summary: pd.DataFrame) -> dict[int, str]:
    return dict(zip(summary["cluster_id"].astype(int), summary["persona"]))
