"""Grouping and summary statistics for metrics."""
from __future__ import annotations

import pandas as pd


def group_stats(results_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Aggregate metric statistics by tag."""

    if results_df.empty:
        return pd.DataFrame(columns=["tag", "mean", "std", "sem", "n"])
    mask = results_df["metric_name"] == metric_name
    metric_df = results_df.loc[mask, ["tag", "value"]].copy()
    grouped = metric_df.groupby("tag", dropna=False)
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = ["mean", "std", "n"]
    summary["sem"] = summary["std"] / summary["n"].pow(0.5)
    summary = summary.reset_index()
    return summary[["tag", "mean", "std", "sem", "n"]]
