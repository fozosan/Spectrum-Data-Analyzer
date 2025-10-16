"""CSV loading utilities for Raman data."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


NUMERIC_COLUMNS = {"center", "height", "area", "fwhm", "area_pct", "peak_index"}
COLUMN_ALIASES = {
    "center": {"center", "centre", "position", "pos", "peak_center"},
    "height": {"height", "intensity", "peak_height"},
    "area": {"area", "peak_area"},
    "fwhm": {"fwhm", "width", "peak_width"},
    "area_pct": {"area_pct", "area%", "area_percent"},
    "peak_id": {"peak_id", "id"},
    "peak_index": {"peak_index", "index"},
    "file": {"file", "filename", "filepath", "path"},
}


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _map_columns(columns: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    normalized = [_normalize_column_name(col) for col in columns]
    for original, norm in zip(columns, normalized):
        for canonical, aliases in COLUMN_ALIASES.items():
            if norm in aliases:
                mapping[original] = canonical
                break
        else:
            mapping[original] = norm
    return mapping


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect canonical column names in the provided DataFrame."""

    mapping = _map_columns(df.columns)
    result: Dict[str, str] = {}
    for original, canonical in mapping.items():
        result.setdefault(canonical, original)
    return result


def _ensure_file_column(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    mapped = _map_columns(df.columns)
    if not any(mapped[col] == "file" for col in df.columns):
        df = df.copy()
        df.insert(0, "file", path.stem)
    else:
        df = df.rename(columns={col: "file" for col in df.columns if mapped[col] == "file"})
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if _normalize_column_name(col) in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_csvs(paths: List[str]) -> pd.DataFrame:
    """Load and concatenate CSV files into a unified dataframe."""

    frames: List[pd.DataFrame] = []
    for path_str in paths:
        path = Path(path_str)
        frame = pd.read_csv(path)
        frame = _ensure_file_column(frame, path)
        frame = frame.rename(columns=_map_columns(frame.columns))
        frame = _coerce_numeric(frame)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def load_csv_tables(paths: Iterable[str]) -> Dict[str, pd.DataFrame]:
    """Load each CSV *as-is* and return a mapping of file identifiers to tables.

    Multiple keys are generated for each file so downstream callers can reuse
    whichever identifier matches their tidy data:

    - Basename including the extension (e.g. ``foo.csv``)
    - Basename stem without extension (e.g. ``foo``)
    - Absolute path to the file
    """

    tables: Dict[str, pd.DataFrame] = {}
    for path_str in paths:
        try:
            df = pd.read_csv(path_str)
        except Exception:
            # Skip unreadable files but keep loading the rest.
            continue
        base = os.path.basename(path_str)
        stem, _ = os.path.splitext(base)
        tables[base] = df
        if stem:
            tables[stem] = df
        tables[os.path.abspath(path_str)] = df
    return tables
