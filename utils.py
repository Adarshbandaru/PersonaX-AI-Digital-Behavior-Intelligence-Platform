"""Utility functions and insight generation for PersonaX AI."""

from __future__ import annotations

import pandas as pd


def format_metric(value, suffix: str = "", fallback: str = "N/A") -> str:
    if value is None:
        return fallback
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def generate_ai_insights(summary: pd.DataFrame, anomaly_cards: dict[str, float | int | str]) -> list[str]:
    insights: list[str] = []
    if summary.empty:
        return ["Not enough cluster structure was found to generate reliable insights."]

    productive = summary.sort_values("productivity_score", ascending=False).iloc[0] if "productivity_score" in summary else None
    if productive is not None:
        insights.append(
            f"Most productive cluster: {productive['persona']} averages {productive['productivity_score']:.1f}/100 productivity with {productive.get('study_time', 0):.1f} study hours."
        )

    risk_frame = summary.copy()
    for col in ["screen_time", "unlock_frequency", "social_media_time", "app_switches"]:
        if col not in risk_frame:
            risk_frame[col] = 0
    risk_frame["addiction_risk"] = (
        risk_frame["screen_time"] * 0.35
        + risk_frame["unlock_frequency"] * 0.015
        + risk_frame["social_media_time"] * 0.018
        + risk_frame["app_switches"] * 0.012
    )
    risky = risk_frame.sort_values("addiction_risk", ascending=False).iloc[0]
    insights.append(
        f"Highest addiction-risk cluster: {risky['persona']} shows elevated screen time, app switching, and checking frequency."
    )

    if {"sleep_hours", "screen_time", "productivity_score"}.issubset(summary.columns):
        balanced = summary.assign(
            balance_score=lambda x: (x["sleep_hours"] - 7.5).abs()
            + (x["screen_time"] - 5.5).abs()
            + ((x["productivity_score"] - 72).abs() / 20)
        ).sort_values("balance_score").iloc[0]
        insights.append(
            f"Most balanced behavioral group: {balanced['persona']} combines {balanced['sleep_hours']:.1f} sleep hours with controlled screen exposure."
        )

    insights.append(
        f"Abnormal usage patterns detected: {anomaly_cards['abnormal_users']} users ({anomaly_cards['abnormal_rate']:.1f}%) were flagged, with {anomaly_cards['highest_risk_user']} showing the highest anomaly score."
    )
    return insights


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
