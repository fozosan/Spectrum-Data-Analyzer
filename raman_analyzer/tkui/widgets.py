"""Reusable Tkinter widgets for the Raman Analyzer UI."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import tkinter as tk
from tkinter import ttk


class ScrollFrame(ttk.Frame):
    """Simple scrollable frame implemented with a ``Canvas``."""

    def __init__(self, master: tk.Misc, *args, **kwargs) -> None:
        super().__init__(master, *args, **kwargs)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")

        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)


class DataTable(ttk.Frame):
    """Lightweight DataFrame viewer with double-click callbacks."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_cell_double_click: Optional[Callable[[int, int, Any], None]] = None,
    ) -> None:
        super().__init__(master)
        self._df = pd.DataFrame()
        self._on_cell_double_click = on_cell_double_click

        self.tree = ttk.Treeview(self, columns=(), show="headings")
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.hscroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.vscroll.set, xscrollcommand=self.hscroll.set)

        self.tree.pack(side="top", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")
        self.hscroll.pack(side="bottom", fill="x")

        self.tree.bind("<Double-1>", self._handle_double_click)

    def set_dataframe(self, dataframe: pd.DataFrame | None) -> None:
        self._df = dataframe.copy() if dataframe is not None else pd.DataFrame()

        for column in self.tree["columns"]:
            self.tree.heading(column, text="")
            self.tree.column(column, width=0)

        columns = [str(c) for c in self._df.columns]
        self.tree["columns"] = columns
        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=110, stretch=True)

        for item_id in self.tree.get_children():
            self.tree.delete(item_id)

        for _, row in self._df.iterrows():
            values = [row.get(column, "") for column in self._df.columns]
            self.tree.insert("", "end", values=values)

    # ------------------------------------------------------------------ events
    def _handle_double_click(self, event: tk.Event) -> None:
        if self._df.empty or self._on_cell_double_click is None:
            return

        item_id = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        if not item_id or not column_id:
            return

        try:
            row_index = self.tree.index(item_id)
            column_index = int(column_id.replace("#", "")) - 1
        except Exception:
            return

        value = self._df.iat[row_index, column_index]
        self._on_cell_double_click(row_index + 1, column_index + 1, value)


def _aggregate(values: List[float], mode: str) -> Optional[float]:
    if not values:
        return None
    if mode == "Sum":
        return float(sum(values))
    return float(sum(values) / len(values))


class SelectionPanel(ttk.Frame):
    """Interactive selection builder mirroring the Qt selection panel."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        on_metrics_updated: Optional[Callable[[pd.DataFrame, pd.DataFrame], None]] = None,
    ) -> None:
        super().__init__(master)
        self.on_metrics_updated = on_metrics_updated
        self.file_to_tag: Dict[str, str] = {}

        # picks[bucket][component][file] -> List[(row, col, value)]
        self._picks: Dict[str, Dict[str, Dict[str, List[Tuple[int, int, float]]]]] = {
            "A": {key: {} for key in ("single", "num", "den", "left", "right")},
            "B": {key: {} for key in ("single", "num", "den", "left", "right")},
        }

        self._mode = tk.StringVar(value="Single")
        self._aggregator = tk.StringVar(value="Mean")
        self._armed = tk.StringVar(value="A.single")

        self._build_controls()
        self._mode.trace_add("write", lambda *_args: self._on_mode_changed())
        self._aggregator.trace_add("write", lambda *_args: self._recompute_and_emit())

    # ------------------------------------------------------------------ public API
    def set_context(self, file_to_tag: Dict[str, str]) -> None:
        self.file_to_tag = dict(file_to_tag or {})
        self._refresh_tables()
        self._recompute_and_emit()

    def add_pick(
        self,
        file_id: str,
        row1: int,
        col1: int,
        value: float,
        *,
        target: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        key = target or self._armed.get()
        bucket, component = key.split(".")
        picks = self._picks[bucket][component].setdefault(file_id, [])
        picks.append((int(row1), int(col1), float(value)))
        if tag is not None:
            self.file_to_tag[str(file_id)] = str(tag)
        self._refresh_tables()
        self._recompute_and_emit()

    def get_mode(self) -> str:
        return self._mode.get()

    # ------------------------------------------------------------------ UI helpers
    def _build_controls(self) -> None:
        manual_frame = ttk.LabelFrame(self, text="Manual Selection")
        manual_frame.pack(side="top", fill="x", padx=6, pady=6)

        mode_box = ttk.Frame(manual_frame)
        mode_box.pack(side="top", fill="x", padx=6, pady=(6, 0))
        ttk.Label(mode_box, text="Mode").pack(side="top", anchor="w")
        ttk.Combobox(
            mode_box,
            textvariable=self._mode,
            values=["Single", "Ratio", "Difference"],
            state="readonly",
        ).pack(side="top", fill="x")

        ttk.Label(mode_box, text="Aggregator").pack(side="top", anchor="w", pady=(6, 0))
        ttk.Combobox(
            mode_box,
            textvariable=self._aggregator,
            values=["Mean", "Sum"],
            state="readonly",
        ).pack(side="top", fill="x")

        armed_frame = ttk.LabelFrame(
            self, text="Armed target — double-click a value in the left grid to add"
        )
        armed_frame.pack(side="top", fill="x", padx=6, pady=6)

        radio_container = ttk.Frame(armed_frame)
        radio_container.pack(side="top", fill="x", padx=6, pady=6)
        self._radios: Dict[str, ttk.Radiobutton] = {}

        for key, label in [
            ("A.single", "A (single)"),
            ("B.single", "B (single)"),
            ("A.num", "A (numerator)"),
            ("A.den", "A (denominator)"),
            ("B.num", "B (numerator)"),
            ("B.den", "B (denominator)"),
            ("A.left", "A (left)"),
            ("A.right", "A (right)"),
            ("B.left", "B (left)"),
            ("B.right", "B (right)"),
        ]:
            rb = ttk.Radiobutton(radio_container, text=label, value=key, variable=self._armed)
            rb.pack(side="top", anchor="w")
            self._radios[key] = rb

        self._update_radio_visibility("Single")

        self.tree_a = ttk.Treeview(
            ttk.LabelFrame(self, text="Selection A — Picks"),
            columns=("file", "tag", "component", "row", "col", "value", "count/agg"),
            show="headings",
            height=8,
        )
        self._configure_pick_tree(self.tree_a)

        self.tree_b = ttk.Treeview(
            ttk.LabelFrame(self, text="Selection B — Picks"),
            columns=("file", "tag", "component", "row", "col", "value", "count/agg"),
            show="headings",
            height=8,
        )
        self._configure_pick_tree(self.tree_b)

        self.preview_a = ttk.Treeview(
            ttk.LabelFrame(self, text="Computed A (per file)"),
            columns=("file", "value"),
            show="headings",
            height=6,
        )
        self._configure_preview_tree(self.preview_a)

        self.preview_b = ttk.Treeview(
            ttk.LabelFrame(self, text="Computed B (per file)"),
            columns=("file", "value"),
            show="headings",
            height=6,
        )
        self._configure_preview_tree(self.preview_b)

    def _configure_pick_tree(self, tree: ttk.Treeview) -> None:
        container = tree.master
        container.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        for index, heading in enumerate(
            ("file", "tag", "component", "row", "col", "value", "count/agg"), start=1
        ):
            tree.heading(f"#{index}", text=heading)
            tree.column(f"#{index}", stretch=True, width=90)
        tree.pack(side="top", fill="both", expand=True)

        button_bar = ttk.Frame(container)
        button_bar.pack(side="top", fill="x")
        if tree is self.tree_a:
            ttk.Button(button_bar, text="Remove Selected (A)", command=lambda: self._remove_selected("A")).pack(side="left", padx=(0, 6))
            ttk.Button(button_bar, text="Clear A", command=lambda: self._clear_bucket("A")).pack(side="left")
        else:
            ttk.Button(button_bar, text="Remove Selected (B)", command=lambda: self._remove_selected("B")).pack(side="left", padx=(0, 6))
            ttk.Button(button_bar, text="Clear B", command=lambda: self._clear_bucket("B")).pack(side="left")

    def _configure_preview_tree(self, tree: ttk.Treeview) -> None:
        container = tree.master
        container.pack(side="top", fill="both", expand=True, padx=6, pady=6)
        tree.heading("#1", text="file")
        tree.heading("#2", text="value")
        tree.column("#1", stretch=True, width=120)
        tree.column("#2", stretch=True, width=120)
        tree.pack(side="top", fill="both", expand=True)

    # ------------------------------------------------------------------ actions
    def _on_mode_changed(self) -> None:
        self._update_radio_visibility(self._mode.get())
        self._refresh_tables()
        self._recompute_and_emit()

    def _update_radio_visibility(self, mode: str) -> None:
        visible = {
            "Single": {"A.single", "B.single"},
            "Ratio": {"A.num", "A.den", "B.num", "B.den"},
            "Difference": {"A.left", "A.right", "B.left", "B.right"},
        }[mode]

        for key, radio in self._radios.items():
            if key in visible:
                radio.pack(side="top", anchor="w")
            else:
                radio.pack_forget()

        if self._armed.get() not in visible:
            self._armed.set(next(iter(visible)))

    def _clear_bucket(self, bucket: str) -> None:
        for component in ("single", "num", "den", "left", "right"):
            self._picks[bucket][component] = {}
        self._refresh_tables()
        self._recompute_and_emit()

    def _remove_selected(self, bucket: str) -> None:
        tree = self.tree_a if bucket == "A" else self.tree_b
        selection = tree.selection()
        if not selection:
            return

        for item_id in selection:
            values = tree.item(item_id, "values")
            if not values:
                continue

            file_id, _tag, component, row_str, col_str, value_str, _desc = values
            try:
                row_index = int(row_str)
                col_index = int(col_str)
                value = float(value_str)
            except Exception:
                continue

            entries = self._picks[bucket][component].get(file_id, [])
            for idx, (r1, c1, val) in enumerate(entries):
                if r1 == row_index and c1 == col_index and abs(val - value) <= 1e-12:
                    del entries[idx]
                    break
            if not entries:
                self._picks[bucket][component].pop(file_id, None)

        self._refresh_tables()
        self._recompute_and_emit()

    # ------------------------------------------------------------------ table data
    def _refresh_tables(self) -> None:
        self._populate_pick_tree(self.tree_a, "A")
        self._populate_pick_tree(self.tree_b, "B")

    def _populate_pick_tree(self, tree: ttk.Treeview, bucket: str) -> None:
        for item_id in tree.get_children():
            tree.delete(item_id)

        rows: List[Tuple[str, str, str, int, int, float, str]] = []
        for component in ("single", "num", "den", "left", "right"):
            components = self._picks[bucket][component]
            for file_id, picks in components.items():
                tag = self.file_to_tag.get(file_id, "")
                values = [value for _r, _c, value in picks]
                descriptor = f"n={len(values)}"
                aggregate = _aggregate(values, self._aggregator.get())
                if aggregate is not None:
                    descriptor += f" / {aggregate:.6g}"
                for row, col, value in picks:
                    rows.append((file_id, tag, component, row, col, float(value), descriptor))

        for row in rows:
            tree.insert("", "end", values=row)

    def _compute_bucket(self, bucket: str) -> pd.DataFrame:
        picks = self._picks[bucket]
        files = set()
        for component in picks.values():
            files.update(component.keys())

        results: List[Tuple[str, float]] = []
        mode = self._mode.get()
        agg_mode = self._aggregator.get()

        for file_id in sorted(files):
            if mode == "Single":
                values = [value for _r, _c, value in picks["single"].get(file_id, [])]
                aggregate = _aggregate(values, agg_mode)
            elif mode == "Ratio":
                numerator = _aggregate(
                    [value for _r, _c, value in picks["num"].get(file_id, [])], agg_mode
                )
                denominator = _aggregate(
                    [value for _r, _c, value in picks["den"].get(file_id, [])], agg_mode
                )
                aggregate = (
                    numerator / denominator
                    if numerator is not None and denominator not in (None, 0)
                    else math.nan
                )
            else:  # Difference
                left_value = _aggregate(
                    [value for _r, _c, value in picks["left"].get(file_id, [])], agg_mode
                )
                right_value = _aggregate(
                    [value for _r, _c, value in picks["right"].get(file_id, [])], agg_mode
                )
                aggregate = (
                    math.nan
                    if left_value is None or right_value is None
                    else left_value - right_value
                )

            results.append((file_id, float(aggregate) if aggregate is not None else math.nan))

        return pd.DataFrame(results, columns=["file", "value"])

    def _fill_preview(self, tree: ttk.Treeview, dataframe: pd.DataFrame) -> None:
        for item_id in tree.get_children():
            tree.delete(item_id)

        if dataframe is None or dataframe.empty:
            return

        for _, row in dataframe.iterrows():
            file_id = str(row.get("file", ""))
            value = row.get("value", math.nan)
            display = "" if pd.isna(value) else f"{float(value):.6g}"
            tree.insert("", "end", values=(file_id, display))

    def _recompute_and_emit(self) -> None:
        a_df = self._compute_bucket("A")
        b_df = self._compute_bucket("B")
        self._fill_preview(self.preview_a, a_df)
        self._fill_preview(self.preview_b, b_df)
        if self.on_metrics_updated is not None:
            self.on_metrics_updated(a_df, b_df)


__all__ = ["DataTable", "ScrollFrame", "SelectionPanel"]
