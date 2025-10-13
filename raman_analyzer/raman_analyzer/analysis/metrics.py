"""Metric computation utilities."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from raman_analyzer.analysis.peaks import match_peaks, sum_attribute
from raman_analyzer.models.selections import PeakSelector


def _per_file(df: pd.DataFrame) -> List[str]:
    if "file" not in df.columns:
        raise ValueError("DataFrame must contain a 'file' column")
    return df["file"].dropna().astype(str).unique().tolist()


def compute_single(df: pd.DataFrame, attr: str, selector: PeakSelector) -> pd.DataFrame:
    """Compute per-file sums for a single attribute."""

    files = _per_file(df)
    results = []
    for file_id in files:
        value = sum_attribute(df, file_id, selector, attr)
        results.append({"file": file_id, "value": value})
    return pd.DataFrame(results)


def compute_ratio(
    df: pd.DataFrame,
    attr: str,
    num_selector: PeakSelector,
    den_selector: PeakSelector,
) -> pd.DataFrame:
    """Compute ratios between two peak selections per file."""

    files = _per_file(df)
    rows = []
    for file_id in files:
        numerator = sum_attribute(df, file_id, num_selector, attr)
        denominator = sum_attribute(df, file_id, den_selector, attr)
        value = np.nan
        if numerator is not None and denominator not in (None, 0):
            if denominator is not None and not np.isclose(denominator, 0.0):
                value = numerator / denominator
        rows.append({"file": file_id, "value": value})
    return pd.DataFrame(rows)


def compute_difference(
    df: pd.DataFrame,
    attr: str,
    a_selector: PeakSelector,
    b_selector: PeakSelector,
) -> pd.DataFrame:
    """Compute differences between two selections per file."""

    files = _per_file(df)
    rows = []
    for file_id in files:
        if attr == "center":
            peaks_a = match_peaks(df, file_id, a_selector)
            peaks_b = match_peaks(df, file_id, b_selector)
            center_a = peaks_a["center"].mean() if not peaks_a.empty else np.nan
            center_b = peaks_b["center"].mean() if not peaks_b.empty else np.nan
            value = center_a - center_b if np.all(np.isfinite([center_a, center_b])) else np.nan
        else:
            val_a = sum_attribute(df, file_id, a_selector, attr)
            val_b = sum_attribute(df, file_id, b_selector, attr)
            if val_a is None or val_b is None:
                value = np.nan
            else:
                value = val_a - val_b
        rows.append({"file": file_id, "value": value})
    return pd.DataFrame(rows)


def assemble_results(
    per_file_values: pd.DataFrame, file_to_tag: Dict[str, str], metric_name: str
) -> pd.DataFrame:
    """Combine per-file values with tagging information."""

    if "file" not in per_file_values.columns or "value" not in per_file_values.columns:
        raise ValueError("per_file_values must contain 'file' and 'value'")
    df = per_file_values.copy()
    df["tag"] = df["file"].map(lambda f: file_to_tag.get(f, ""))
    df["metric_name"] = metric_name
    df = df.rename(columns={"value": "value"})
    return df[["file", "tag", "metric_name", "value"]]
