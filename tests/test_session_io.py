from __future__ import annotations

import pandas as pd

from raman_analyzer.io.session_io import session_from_dict, session_to_dict
from raman_analyzer.models.session import AnalysisSession


def test_session_roundtrip_dict() -> None:
    session = AnalysisSession()
    df = pd.DataFrame(
        {
            "file": ["a", "a", "b"],
            "peak_index": [0, 1, 0],
            "area": [10.0, 5.0, 8.0],
            "center": [100.0, 200.0, 110.0],
        }
    )
    session.set_raw_data(df)
    session.set_tag("a", "T1")
    session.set_tag("b", "T2")
    session.update_x_mapping({"a": 1.0, "b": 2.0})
    metric_df = pd.DataFrame({"file": ["a", "b"], "value": [15.0, 9.0]})
    session.update_metric("area_sum", metric_df)
    session.data_fit = {"model": "linear", "coeffs": (1.0, 2.0), "r2": 0.99}
    session.literature_fit = {"model": "linear", "coeffs": (0.5, 1.0)}
    session.intersections = [(0.5, 0.6)]
    session.plot_config = {
        "x_axis": "Custom X (per file)",
        "y_axis": "area_sum",
        "plot_type": "Scatter",
        "error_mode": "SEM",
        "x_limits": (0.0, 10.0),
        "y_limits": (None, None),
    }

    payload = session_to_dict(session)
    restored = session_from_dict(payload)

    pd.testing.assert_frame_equal(
        restored.raw_df.reset_index(drop=True), df.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        restored.results_df.sort_index(axis=1),
        session.results_df.sort_index(axis=1),
        check_dtype=False,
    )
    assert restored.file_to_tag == session.file_to_tag
    assert restored.x_mapping == session.x_mapping
    assert restored.data_fit == session.data_fit
    assert restored.literature_fit == session.literature_fit
    assert restored.intersections == session.intersections
    assert restored.plot_config == session.plot_config
