"""Tests for peak selection helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from raman_analyzer.analysis.peaks import match_peaks
from raman_analyzer.models.selections import PeakSelector


def test_select_by_center_handles_nan() -> None:
    df = pd.DataFrame(
        {
            "file": ["a", "a", "a"],
            "center": [np.nan, 200.0, np.nan],
            "peak_index": [0, 1, 2],
            "area": [1, 1, 1],
        }
    )
    sel = PeakSelector(mode="nearest_center", centers=[200.0], tolerance_cm1=5.0)
    peaks = match_peaks(df, "a", sel)
    assert not peaks.empty and int(peaks.iloc[0]["peak_index"]) == 1

    df2 = df.copy()
    df2["center"] = [np.nan, np.nan, np.nan]
    peaks2 = match_peaks(df2, "a", sel)
    assert peaks2.empty
