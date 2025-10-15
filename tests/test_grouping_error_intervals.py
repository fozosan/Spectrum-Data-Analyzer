from __future__ import annotations

import math
import pandas as pd

from raman_analyzer.analysis.grouping import compute_error_table


def build_xy_df() -> pd.DataFrame:
    """Create a small dataframe with two groups for error interval testing."""
    data = {
        "tag": ["T1", "T1", "T1", "T1", "T1", "T1"],
        "x": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        "y": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
    }
    return pd.DataFrame(data)


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def test_error_table_sd() -> None:
    df = build_xy_df()
    out = compute_error_table(df, mode="SD")
    row1 = out[(out["tag"] == "T1") & (out["x"] == 1.0)].iloc[0]
    row2 = out[(out["tag"] == "T1") & (out["x"] == 2.0)].iloc[0]

    assert approx(row1["mean"], 2.0)
    assert approx(row1["std"], 1.0)
    assert row1["count"] == 3
    assert approx(row1["yerr"], 1.0)
    assert approx(row2["yerr"], 2.0)


def test_error_table_sem() -> None:
    df = build_xy_df()
    out = compute_error_table(df, mode="SEM")
    row1 = out[(out["tag"] == "T1") & (out["x"] == 1.0)].iloc[0]
    row2 = out[(out["tag"] == "T1") & (out["x"] == 2.0)].iloc[0]
    expected_sem1 = 1.0 / math.sqrt(3)
    expected_sem2 = 2.0 / math.sqrt(3)

    assert approx(row1["sem"], expected_sem1)
    assert approx(row1["yerr"], expected_sem1)
    assert approx(row2["sem"], expected_sem2)
    assert approx(row2["yerr"], expected_sem2)


def test_error_table_ci95() -> None:
    df = build_xy_df()
    out = compute_error_table(df, mode="95% CI")
    row1 = out[(out["tag"] == "T1") & (out["x"] == 1.0)].iloc[0]
    row2 = out[(out["tag"] == "T1") & (out["x"] == 2.0)].iloc[0]
    tcrit = 4.303  # t critical for sample size n=3
    expected1 = tcrit * (1.0 / math.sqrt(3))
    expected2 = tcrit * (2.0 / math.sqrt(3))

    assert approx(row1["yerr"], expected1, tol=1e-3)
    assert approx(row2["yerr"], expected2, tol=1e-3)
