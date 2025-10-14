"""Grouping helpers and summary statistics for metrics."""
from __future__ import annotations

import numpy as np
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


def _t_critical_95(n: int) -> float:
    """Two-tailed t critical value at 95% CI for sample size ``n`` (df = n - 1).

    Returns 0 for ``n`` ≤ 1 and approaches 1.96 for large ``n``.
    """

    if n <= 1:
        return 0.0
    # Keys are sample sizes (n), not df.
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    if n <= 30:
        return table.get(n, 2.045)
    return 1.96


def compute_error_table(df_xy: pd.DataFrame, mode: str = "None") -> pd.DataFrame:
    """Compute grouped error intervals for scatter-style data frames.

    Parameters
    ----------
    df_xy:
        DataFrame expected to contain ``tag``, ``x`` and ``y`` columns.
    mode:
        One of ``"None"``, ``"SD"``, ``"SEM"`` or ``"95% CI"``.

    Returns a DataFrame with the columns ``tag``, ``x``, ``mean``, ``std``,
    ``count``, ``sem``, ``ci95`` and ``yerr`` (matching the requested mode).
    """

    required = {"tag", "x", "y"}
    if not required.issubset(df_xy.columns):
        return pd.DataFrame(columns=["tag", "x", "mean", "std", "count", "sem", "ci95", "yerr"])

    grouped = (
        df_xy.groupby(["tag", "x"], dropna=False)["y"].agg(["mean", "std", "count"]).reset_index()
    )
    if grouped.empty:
        grouped["sem"] = np.nan
        grouped["ci95"] = np.nan
        grouped["yerr"] = np.nan
        return grouped

    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].where(grouped["count"] > 0, np.nan))

    ci_values: list[float] = []
    for count, sem in zip(grouped["count"].to_numpy(dtype=int), grouped["sem"].to_numpy(dtype=float)):
        if count < 2 or not np.isfinite(sem):
            ci_values.append(0.0)
        else:
            ci_values.append(_t_critical_95(int(count)) * float(sem))
    grouped["ci95"] = np.array(ci_values, dtype=float)

    if mode == "SD":
        grouped["yerr"] = grouped["std"].fillna(0.0)
    elif mode == "SEM":
        grouped["yerr"] = grouped["sem"].fillna(0.0)
    elif mode == "95% CI":
        grouped["yerr"] = grouped["ci95"].fillna(0.0)
    else:
        grouped["yerr"] = 0.0

    return grouped
