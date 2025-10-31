"""Application session state management."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class AnalysisSession:
    """In-memory representation of an analysis session.

    Attributes
    ----------
    raw_df:
        Raw peak data aggregated from all loaded CSV files.
    file_to_tag:
        Mapping of file identifiers to user-assigned group tags.
    results_df:
        Wide-format table containing per-file metric results.
    ordering:
        Optional mapping of files to user-provided ordering values.
    data_fit:
        Metadata for the fitted trendline of the current data.
    literature_fit:
        Metadata for an optional literature trendline overlay.
    intersections:
        Points where the data and literature fits intersect.
    """

    raw_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    file_to_tag: Dict[str, str] = field(default_factory=dict)
    results_df: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["file", "tag"])
    )
    ordering: Dict[str, float] = field(default_factory=dict)
    data_fit: Optional[dict] = None
    literature_fit: Optional[dict] = None
    intersections: List[tuple[float, float]] = field(default_factory=list)
    raw_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Persisted Selection Panel state (mode, aggregator, picks)
    selection_state: Optional[dict] = None

    def ensure_files(self, files: Iterable[str]) -> None:
        """Ensure that all files exist in :attr:`results_df`.

        Parameters
        ----------
        files:
            Iterable of file identifiers to add to the results table if missing.
        """

        if self.results_df.empty:
            self.results_df = pd.DataFrame({"file": list(files)})
            self.results_df["tag"] = [self.file_to_tag.get(f, "") for f in files]
            return

        existing = set(self.results_df["file"])
        new_files = [f for f in files if f not in existing]
        if new_files:
            additions = pd.DataFrame({"file": new_files})
            additions["tag"] = [self.file_to_tag.get(f, "") for f in new_files]
            self.results_df = pd.concat([self.results_df, additions], ignore_index=True)
        self.results_df["tag"] = self.results_df["file"].map(
            lambda f: self.file_to_tag.get(f, "")
        )

    def set_raw_data(self, df: pd.DataFrame) -> None:
        """Assign raw data and refresh dependent state."""

        self.raw_df = df.copy()
        files = self.raw_df["file"].unique().tolist() if not df.empty else []
        self.ensure_files(files)
        self.data_fit = None
        self.literature_fit = None
        self.intersections.clear()

    def set_tag(self, file_id: str, tag: str) -> None:
        """Assign a tag to a file and update the results table."""

        self.file_to_tag[file_id] = tag
        if "file" in self.results_df.columns:
            mask = self.results_df["file"] == file_id
            if mask.any():
                self.results_df.loc[mask, "tag"] = tag
            else:
                self.ensure_files([file_id])
        else:
            self.ensure_files([file_id])

    # ------------------------------------------------------------------ raw tables
    def set_raw_tables(self, tables: Dict[str, pd.DataFrame]) -> None:
        """Store non-normalized CSV tables keyed by file identifier."""

        self.raw_tables = dict(tables or {})

    def get_raw_table(self, file_id: str) -> Optional[pd.DataFrame]:
        """Retrieve a raw table matching ``file_id`` if available."""

        if not self.raw_tables:
            return None
        if file_id in self.raw_tables:
            return self.raw_tables[file_id]

        stem, _ = os.path.splitext(file_id)
        if stem and stem in self.raw_tables:
            return self.raw_tables[stem]

        for key, table in self.raw_tables.items():
            k_stem, _ = os.path.splitext(key)
            if k_stem == stem and table is not None:
                return table
        return None

    def update_metric(self, metric_name: str, values_df: pd.DataFrame) -> None:
        """Merge a metric column into :attr:`results_df`.

        Parameters
        ----------
        metric_name:
            Name of the metric column to insert/update.
        values_df:
            DataFrame with columns ``file`` and ``value``.
        """

        if "file" not in values_df.columns or "value" not in values_df.columns:
            raise ValueError("values_df must contain 'file' and 'value' columns")

        self.ensure_files(values_df["file"].tolist())
        if metric_name in self.results_df.columns:
            self.results_df = self.results_df.drop(columns=[metric_name])
        values_df = values_df.rename(columns={"value": metric_name})
        self.results_df = self.results_df.merge(values_df, on="file", how="left")

    def update_ordering(self, mapping: dict[str, float]) -> None:
        """Replace the entire Ordering map."""

        self.ordering = dict(mapping or {})

    def update_x_mapping(self, *_args, **_kwargs):
        raise NotImplementedError(
            "update_x_mapping() was removed. Use update_ordering(mapping) and X='Ordering'."
        )

    def invalidate_fits(self) -> None:
        """Clear cached fit data."""

        self.data_fit = None
        self.intersections.clear()

    def set_data_fit(self, fit: Optional[dict]) -> None:
        """Store the latest data fit and reset intersections."""

        self.data_fit = fit
        self.intersections.clear()

    def set_literature_fit(self, fit: Optional[dict]) -> None:
        """Store the literature fit and reset intersections."""

        self.literature_fit = fit
        self.intersections.clear()

    def set_intersections(self, points: List[tuple[float, float]]) -> None:
        """Persist intersection points for later reference."""

        self.intersections = points
