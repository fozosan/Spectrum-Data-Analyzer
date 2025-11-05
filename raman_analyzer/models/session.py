"""Application session state management."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import pandas as pd


APPEND_ONLY = True


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
    # Index of per-TXT subsets: key = value from CSV's 'file' column (e.g., "...pos1.txt")
    raw_tables_by_file: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Persisted Selection Panel state (mode, aggregator, picks)
    selection_state: Optional[dict] = None
    # ---- Inverse-solve memory (flat; no nested configs passed to runner)
    inv_params_linear: Dict[str, float] = field(default_factory=dict)
    inv_params_quadratic: Dict[str, float] = field(default_factory=dict)
    inv_params_power: Dict[str, float] = field(default_factory=dict)
    inv_model_last: Optional[str] = None
    inv_y_metric_last: Optional[str] = None
    inv_y_source_last: Optional[str] = None
    inv_plot_on_chart_last: Optional[bool] = None

    # ---- Helpers (remain purely UI/session-scoped)
    def get_inv_params(self, model: str) -> Dict[str, float]:
        m = (model or "").strip().lower()
        if m == "linear":
            return dict(self.inv_params_linear)
        if m == "quadratic":
            return dict(self.inv_params_quadratic)
        if m == "power":
            return dict(self.inv_params_power)
        return {}

    def set_inv_params(self, model: str, params: Dict[str, float]) -> None:
        if not model:
            return
        # sanitize to floats
        clean: Dict[str, float] = {}
        for k, v in (params or {}).items():
            try:
                clean[str(k)] = float(v)
            except Exception:
                continue
        m = model.strip().lower()
        if m == "linear":
            self.inv_params_linear = clean
        elif m == "quadratic":
            self.inv_params_quadratic = clean
        elif m == "power":
            self.inv_params_power = clean

    def has_files(self) -> bool:
        """Return ``True`` if any raw tables are loaded."""

        return bool(self.raw_tables)

    def list_files(self) -> List[str]:
        """Return known file identifiers in load order."""

        if not self.results_df.empty and "file" in self.results_df.columns:
            files = self.results_df["file"].astype(str).tolist()
            seen: set[str] = set()
            ordered: List[str] = []
            for file_id in files:
                if not file_id or file_id in seen:
                    continue
                seen.add(file_id)
                ordered.append(file_id)
            return ordered
        if self.raw_tables:
            ordered: List[str] = []
            seen_tables: set[str] = set()
            for key in self.raw_tables.keys():
                if not key or key in seen_tables:
                    continue
                seen_tables.add(key)
                ordered.append(key)
            return ordered
        if not self.raw_df.empty and "file" in self.raw_df.columns:
            files = self.raw_df["file"].astype(str).dropna().tolist()
            seen_raw: set[str] = set()
            ordered_raw: List[str] = []
            for file_id in files:
                if not file_id or file_id in seen_raw:
                    continue
                seen_raw.add(file_id)
                ordered_raw.append(file_id)
            return ordered_raw
        return []

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
        """Add raw data without discarding previously loaded files."""

        if df is None or df.empty:
            return

        incoming = df.copy()
        if "file" not in incoming.columns:
            raise ValueError("Raw data must include a 'file' column for additive imports.")

        incoming["file"] = incoming["file"].astype(str)
        existing_files: set[str] = set()
        if not self.raw_df.empty and "file" in self.raw_df.columns:
            existing_files = set(self.raw_df["file"].astype(str).tolist())

        # Skip any rows whose file id already exists; additive only.
        append_mask = ~incoming["file"].isin(existing_files)
        append_df = incoming.loc[append_mask].copy()
        if append_df.empty:
            return

        if self.raw_df is None or self.raw_df.empty:
            self.raw_df = append_df
        else:
            self.raw_df = pd.concat([self.raw_df, append_df], ignore_index=True, sort=False)

        new_files = append_df["file"].dropna().unique().tolist()
        if new_files:
            self.ensure_files(new_files)

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

        if not tables:
            return

        for file_id, table in tables.items():
            key = str(file_id)
            if not key or key in self.raw_tables:
                continue
            self.raw_tables[key] = table
            # Build per-file index so Auto-populate can resolve TXT names from CSV content
            try:
                if table is not None and not table.empty and "file" in table.columns:
                    col = table["file"].astype(str)
                    for fid in col.dropna().unique():
                        fid_str = str(fid)
                        if not fid_str or fid_str in self.raw_tables_by_file:
                            continue
                        subset = table.loc[col == fid_str].copy()
                        self.raw_tables_by_file[fid_str] = subset
            except Exception:
                # Keep additive behavior; don't raise from UI callback.
                pass

    def get_raw_table(self, file_id: str) -> Optional[pd.DataFrame]:
        """Retrieve a raw table matching ``file_id`` if available."""

        if not self.raw_tables:
            return None
        # 0) Exact hit on per-TXT subsets (preferred)
        if file_id in self.raw_tables_by_file:
            return self.raw_tables_by_file[file_id]

        # Helper for case-insensitive basename/root comparison
        def _base_and_root(s: str) -> tuple[str, str]:
            s = str(s or "")
            base = os.path.basename(s)
            root, _ = os.path.splitext(base)
            return base.lower(), root.lower()

        tgt_base, tgt_root = _base_and_root(file_id)

        # 1) Fuzzy hit on per-TXT subsets by basename or root (handles path vs bare name)
        if tgt_base or tgt_root:
            for k, tbl in self.raw_tables_by_file.items():
                base_k, root_k = _base_and_root(k)
                if base_k == tgt_base or (tgt_root and root_k == tgt_root):
                    return tbl

        # 2) Fallback: exact key to the whole CSV table (rarely used by Auto-populate)
        if file_id in self.raw_tables:
            return self.raw_tables[file_id]

        # 3) Fuzzy: map by basename/root to a whole CSV table if nothing else matched
        if tgt_base or tgt_root:
            for key, table in self.raw_tables.items():
                base_k, root_k = _base_and_root(key)
                if base_k == tgt_base or (tgt_root and root_k == tgt_root):
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
