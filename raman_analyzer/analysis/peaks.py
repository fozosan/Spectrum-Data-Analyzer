"""Utilities for peak selection and aggregation."""
from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from raman_analyzer.models.selections import PeakSelector


def _select_by_index(df: pd.DataFrame, indices: List[int]) -> pd.DataFrame:
    if not indices:
        return df.iloc[0:0]
    if "peak_index" in df.columns:
        return df[df["peak_index"].isin(indices)]
    valid = [i for i in indices if 0 <= i < len(df)]
    if not valid:
        return df.iloc[0:0]
    return df.iloc[valid]


def _select_by_center(df: pd.DataFrame, centers: List[float], tolerance: float) -> pd.DataFrame:
    if "center" not in df.columns or not centers:
        return df.iloc[0:0]
    selected_indices: List[int] = []
    centers_series = pd.to_numeric(df["center"], errors="coerce")
    for target in centers:
        diffs = (centers_series - float(target)).abs()
        finite = diffs[np.isfinite(diffs)]
        if finite.empty:
            continue
        min_idx = finite.idxmin()
        if finite.loc[min_idx] <= tolerance:
            selected_indices.append(min_idx)
    if not selected_indices:
        return df.iloc[0:0]
    return df.loc[selected_indices].drop_duplicates()


def match_peaks(df: pd.DataFrame, file_id: str, selector: PeakSelector) -> pd.DataFrame:
    """Return peaks matching the selector for a given file."""

    slice_df = df[df["file"] == file_id]
    if slice_df.empty:
        return slice_df
    if selector.is_by_index():
        return _select_by_index(slice_df, selector.indices)
    return _select_by_center(slice_df, selector.centers, selector.tolerance_cm1)


def sum_attribute(
    df: pd.DataFrame, file_id: str, selector: PeakSelector, attr: str
) -> Optional[float]:
    """Sum the provided attribute across selected peaks."""

    peaks = match_peaks(df, file_id, selector)
    if peaks.empty or attr not in peaks.columns:
        return None
    values = pd.to_numeric(peaks[attr], errors="coerce")
    if values.isna().all():
        return None
    return float(values.sum(skipna=True))


def aggregate_attribute(
    df: pd.DataFrame,
    file_id: str,
    selector: PeakSelector,
    attr: str,
    agg: Literal["sum", "mean"] = "sum",
) -> Optional[float]:
    """Aggregate an attribute across selected peaks with a chosen reducer."""

    peaks = match_peaks(df, file_id, selector)
    if peaks.empty or attr not in peaks.columns:
        return None
    values = pd.to_numeric(peaks[attr], errors="coerce").dropna()
    if values.empty:
        return None
    agg = (agg or "sum").lower()
    if agg == "mean":
        return float(values.mean())
    return float(values.sum())
