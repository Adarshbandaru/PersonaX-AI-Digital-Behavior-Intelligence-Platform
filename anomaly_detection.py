"""Anomaly detection models for PersonaX AI."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler


def detect_anomalies(
    scaled_df: pd.DataFrame,
    method: str = "Isolation Forest",
    contamination: float = 0.06,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, object]:
    """Return anomaly labels and normalized anomaly scores. -1 means abnormal."""
    X = scaled_df.values
    if method == "Isolation Forest":
        model = IsolationForest(contamination=contamination, random_state=random_state)
        labels = model.fit_predict(X)
        raw_scores = -model.decision_function(X)
    else:
        n_neighbors = min(35, max(5, len(scaled_df) // 12))
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        labels = model.fit_predict(X)
        raw_scores = -model.negative_outlier_factor_

    normalized = MinMaxScaler(feature_range=(0, 100)).fit_transform(raw_scores.reshape(-1, 1)).ravel()
    return labels, normalized, model


def anomaly_score_cards(df: pd.DataFrame, scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int | str]:
    abnormal_count = int((labels == -1).sum())
    top_idx = int(np.argmax(scores))
    return {
        "abnormal_users": abnormal_count,
        "abnormal_rate": abnormal_count / max(len(df), 1) * 100,
        "highest_score": float(scores[top_idx]),
        "highest_risk_user": str(df.iloc[top_idx]["user_id"]),
    }
