"""Serialization helpers for saving and restoring analysis sessions."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd

from raman_analyzer.models.session import AnalysisSession


def _to_records(df: Optional[pd.DataFrame]) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def session_to_dict(session: AnalysisSession) -> Dict[str, Any]:
    """Serialize a session to a JSON-safe dictionary."""

    raw_df = getattr(session, "raw_df", pd.DataFrame())
    results_df = getattr(session, "results_df", pd.DataFrame())
    file_to_tag = dict(getattr(session, "file_to_tag", {}) or {})
    ordering = dict(getattr(session, "ordering", {}) or {})
    data_fit = getattr(session, "data_fit", None)
    literature_fit = getattr(session, "literature_fit", None)
    intersections = getattr(session, "intersections", []) or []
    plot_config = getattr(session, "plot_config", None)

    payload: Dict[str, Any] = {
        "version": 1,
        "raw_df": _to_records(raw_df),
        "results_df": _to_records(results_df),
        "file_to_tag": file_to_tag,
        "ordering": {k: float(v) for k, v in dict(ordering).items()},
        "data_fit": data_fit,
        "literature_fit": literature_fit,
        "plot_config": plot_config,  # Tk/Qt share this persisted config
        "intersections": [
            {"x": float(point[0]), "y": float(point[1])}
            for point in intersections
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ],
    }
    selection_state = getattr(session, "selection_state", None)
    if isinstance(selection_state, dict):
        payload["selection_state"] = selection_state
    return payload


def session_from_dict(data: Dict[str, Any]) -> AnalysisSession:
    """Rebuild a session from serialized data."""

    sess = AnalysisSession()

    raw_df = pd.DataFrame(data.get("raw_df", []))
    if not raw_df.empty:
        sess.set_raw_data(raw_df)

    for file_id, tag in (data.get("file_to_tag", {}) or {}).items():
        sess.set_tag(str(file_id), str(tag))

    ordering_map = data.get("ordering")
    if ordering_map is None:
        if data.get("x_mapping"):
            raise ValueError(
                "Session payload is missing 'ordering'. Use tools/migrate_session_ordering.py to upgrade."
            )
        ordering_map = {}
    if ordering_map:
        sess.update_ordering({str(k): float(v) for k, v in dict(ordering_map).items()})

    results_df = pd.DataFrame(data.get("results_df", []))
    if not results_df.empty:
        sess.results_df = results_df

    sess.data_fit = data.get("data_fit") or None
    sess.literature_fit = data.get("literature_fit") or None
    sess.plot_config = data.get("plot_config") or {}
    sel = data.get("selection_state")
    if isinstance(sel, dict):
        sess.selection_state = sel

    intersections = data.get("intersections", []) or []
    points: list[tuple[float, float]] = []
    for item in intersections:
        if isinstance(item, dict):
            try:
                x_val = float(item["x"])
                y_val = float(item["y"])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((x_val, y_val))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    sess.intersections = points

    return sess


def save_session(path: str, session: AnalysisSession) -> None:
    payload = session_to_dict(session)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)


def load_session(path: str) -> AnalysisSession:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return session_from_dict(data)
