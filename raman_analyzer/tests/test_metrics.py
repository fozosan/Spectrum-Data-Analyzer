"""Unit tests for metric computations."""
from __future__ import annotations

import pandas as pd

from raman_analyzer.analysis.metrics import (
    compute_difference,
    compute_ratio,
    compute_single,
)
from raman_analyzer.models.selections import PeakSelector


def build_mock_df() -> pd.DataFrame:
    data = {
        "file": ["a", "a", "a", "b", "b", "b"],
        "peak_index": [0, 1, 2, 0, 1, 2],
        "area": [10.0, 5.0, 2.5, 8.0, 4.0, 1.0],
        "center": [100.0, 200.0, 300.0, 110.0, 205.0, 310.0],
    }
    return pd.DataFrame(data)


def test_compute_single_sum() -> None:
    df = build_mock_df()
    selector = PeakSelector(mode="by_index", indices=[0, 2])
    result = compute_single(df, "area", selector)
    expected = {
        "a": 12.5,
        "b": 9.0,
    }
    for _, row in result.iterrows():
        assert row["value"] == expected[row["file"]]


def test_compute_single_mean() -> None:
    df = build_mock_df()
    selector = PeakSelector(mode="by_index", indices=[0, 2])
    result = compute_single(df, "area", selector, agg="mean")
    expected = {
        "a": 6.25,
        "b": 4.5,
    }
    for _, row in result.iterrows():
        assert row["value"] == expected[row["file"]]


def test_compute_ratio() -> None:
    df = build_mock_df()
    num = PeakSelector(mode="by_index", indices=[0])
    den = PeakSelector(mode="by_index", indices=[1])
    result = compute_ratio(df, "area", num, den)
    expected = {
        "a": 2.0,
        "b": 2.0,
    }
    for _, row in result.iterrows():
        assert row["value"] == expected[row["file"]]


def test_compute_ratio_mean() -> None:
    df = build_mock_df()
    num = PeakSelector(mode="by_index", indices=[0, 2])
    den = PeakSelector(mode="by_index", indices=[1])
    result = compute_ratio(df, "area", num, den, agg="mean")
    expected = {
        "a": 6.25 / 5.0,
        "b": 4.5 / 4.0,
    }
    for _, row in result.iterrows():
        assert row["value"] == expected[row["file"]]


def test_compute_difference_center() -> None:
    df = build_mock_df()
    sel_a = PeakSelector(mode="by_index", indices=[1])
    sel_b = PeakSelector(mode="by_index", indices=[0])
    result = compute_difference(df, "center", sel_a, sel_b)
    expected = {
        "a": 100.0,
        "b": 95.0,
    }
    for _, row in result.iterrows():
        assert row["value"] == expected[row["file"]]
