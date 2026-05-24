"""Data generation and preprocessing utilities for PersonaX AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES = [
    "screen_time",
    "unlock_frequency",
    "app_switches",
    "social_media_time",
    "gaming_time",
    "study_time",
    "sleep_hours",
    "productivity_score",
]


@dataclass
class PreprocessResult:
    clean_df: pd.DataFrame
    scaled_df: pd.DataFrame
    scaler: StandardScaler
    selected_features: list[str]


def generate_synthetic_behavior_data(n_users: int = 420, random_state: int = 42) -> pd.DataFrame:
    """Create realistic synthetic digital behavior data for a polished demo."""
    rng = np.random.default_rng(random_state)
    segments = rng.choice(
        ["focus", "balanced", "night_scroll", "switcher", "gamer"],
        size=n_users,
        p=[0.22, 0.28, 0.18, 0.18, 0.14],
    )

    profiles = {
        "focus": (5.4, 55, 62, 55, 22, 5.8, 7.4, 86),
        "balanced": (4.8, 64, 75, 70, 35, 3.4, 7.8, 74),
        "night_scroll": (8.7, 118, 138, 190, 48, 1.8, 5.5, 42),
        "switcher": (7.2, 145, 210, 145, 65, 2.4, 6.2, 50),
        "gamer": (7.8, 92, 125, 85, 165, 2.0, 6.7, 48),
    }
    spreads = np.array([0.85, 18, 28, 34, 24, 0.9, 0.55, 8])

    rows = []
    for idx, segment in enumerate(segments, start=1):
        base = np.array(profiles[segment], dtype=float)
        values = rng.normal(base, spreads)
        screen_time, unlocks, switches, social, gaming, study, sleep, productivity = values

        rows.append(
            {
                "user_id": f"PX-{idx:04d}",
                "screen_time": np.clip(screen_time, 1.6, 13.8),
                "unlock_frequency": int(np.clip(unlocks, 12, 230)),
                "app_switches": int(np.clip(switches, 20, 340)),
                "social_media_time": int(np.clip(social, 5, 320)),
                "gaming_time": int(np.clip(gaming, 0, 260)),
                "study_time": np.clip(study, 0.1, 9.5),
                "sleep_hours": np.clip(sleep, 3.8, 9.6),
                "productivity_score": int(np.clip(productivity, 12, 98)),
                "device_type": rng.choice(["Android", "iOS", "Tablet"], p=[0.58, 0.35, 0.07]),
                "primary_context": rng.choice(["Student", "Remote learner", "Hybrid worker"], p=[0.62, 0.23, 0.15]),
            }
        )

    df = pd.DataFrame(rows)

    anomaly_count = max(8, n_users // 25)
    anomaly_indices = rng.choice(df.index, anomaly_count, replace=False)
    df.loc[anomaly_indices, "screen_time"] = rng.uniform(10.5, 15.5, anomaly_count)
    df.loc[anomaly_indices, "unlock_frequency"] = rng.integers(170, 285, anomaly_count)
    df.loc[anomaly_indices, "sleep_hours"] = rng.uniform(3.3, 5.1, anomaly_count)
    df.loc[anomaly_indices, "productivity_score"] = rng.integers(8, 38, anomaly_count)

    return df


def ensure_user_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a usable user identifier column for hover labels and downloads."""
    output = df.copy()
    if "user_id" not in output.columns:
        output.insert(0, "user_id", [f"USER-{i:04d}" for i in range(1, len(output) + 1)])
    return output


def available_numeric_features(df: pd.DataFrame, preferred: Iterable[str] = DEFAULT_FEATURES) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    preferred_available = [col for col in preferred if col in numeric_columns]
    extra_available = [col for col in numeric_columns if col not in preferred_available]
    return preferred_available + extra_available


def preprocess_dataset(df: pd.DataFrame, selected_features: list[str]) -> PreprocessResult:
    """Handle missing values and standardize selected numeric columns."""
    if not selected_features:
        raise ValueError("Select at least one numeric feature for analysis.")

    clean_df = ensure_user_id(df)
    for feature in selected_features:
        clean_df[feature] = pd.to_numeric(clean_df[feature], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(clean_df[selected_features])
    clean_df[selected_features] = imputed_values

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(clean_df[selected_features])
    scaled_df = pd.DataFrame(scaled_values, columns=selected_features, index=clean_df.index)

    return PreprocessResult(clean_df=clean_df, scaled_df=scaled_df, scaler=scaler, selected_features=selected_features)
