"""Reusable Tkinter widgets for the Raman Analyzer UI."""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk


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
        session: Any | None = None,
        on_metrics: Optional[
            Callable[[str, pd.DataFrame, str, pd.DataFrame], None]
        ] = None,
        on_autopopulate: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> None:
        super().__init__(master)
        self.session = session  # for API parity; not used directly here
        self._cb_metrics = on_metrics
        self._cb_autopop = on_autopopulate
        self.file_to_tag: Dict[str, str] = {}

        # picks[bucket][component][file] -> List[(row, col, value)]
        self._picks: Dict[str, Dict[str, Dict[str, List[Tuple[int, int, float]]]]] = {
            "A": {key: {} for key in ("single", "num", "den", "left", "right")},
            "B": {key: {} for key in ("single", "num", "den", "left", "right")},
        }

        self._mode = tk.StringVar(value="Single")
        self._aggregator = tk.StringVar(value="Mean")
        self._armed: str = "A.single"
        self._armed_var = tk.StringVar(value=self._armed)

        self._build_controls()
        self._mode.trace_add("write", lambda *_args: self._on_mode_changed())
        self._aggregator.trace_add("write", lambda *_args: self._refresh_tables_and_emit())
        self._armed_var.trace_add("write", lambda *_: self._set_armed(self._armed_var.get()))

    # ------------------------------------------------------------------ public API
    def set_context(self, file_to_tag: Dict[str, str]) -> None:
        self.file_to_tag = dict(file_to_tag or {})
        self._refresh_tables_and_emit()

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
        key = target or self._armed
        bucket, component = key.split(".")
        picks = self._picks[bucket][component].setdefault(file_id, [])
        picks.append((int(row1), int(col1), float(value)))
        if tag is not None:
            self.file_to_tag[str(file_id)] = str(tag)
        self._refresh_tables_and_emit()

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

        self.targets_box = ttk.LabelFrame(
            self, text="Armed target — double-click a value in the left grid to add"
        )
        self.targets_box.pack(side="top", fill="x", padx=6, pady=6)

        self.targets_panel = ttk.Frame(self.targets_box)
        self.targets_panel.pack(side="top", fill="x", padx=6, pady=6)
        self._target_radios: List[ttk.Radiobutton] = []

        self._help_label = ttk.Label(
            self.targets_box,
            text="Double-click a value in the data table to add it to the armed target.",
            wraplength=320,
            justify="left",
        )
        self._help_label.pack(side="top", fill="x", padx=6, pady=(0, 4))

        autopop_box = ttk.LabelFrame(self, text="Auto-populate from grid")
        autopop_box.pack(side="top", fill="x", padx=6, pady=6)

        form = ttk.Frame(autopop_box)
        form.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Label(form, text="Row").grid(row=0, column=0, sticky="w")
        self.row_var = tk.StringVar(value="1")
        ttk.Entry(form, textvariable=self.row_var, width=6).grid(row=0, column=1, padx=(4, 12))
        ttk.Label(form, text="Column").grid(row=0, column=2, sticky="w")
        self.col_var = tk.StringVar(value="1")
        ttk.Entry(form, textvariable=self.col_var, width=6).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(form, text="Scope").grid(row=0, column=4, sticky="w")
        self.scope_var = tk.StringVar(value="Selected")
        ttk.Combobox(
            form,
            textvariable=self.scope_var,
            values=["Selected", "All"],
            state="readonly",
            width=10,
        ).grid(row=0, column=5, padx=(4, 0))
        ttk.Button(autopop_box, text="Auto-populate", command=self._on_autopop).pack(
            side="top", anchor="w", padx=6, pady=(0, 6)
        )

        self._rebuild_radios()

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
        self._rebuild_radios()
        self._refresh_tables_and_emit()

    def _rebuild_radios(self) -> None:
        for widget in list(self.targets_panel.winfo_children()):
            widget.destroy()
        self._target_radios = []

        mode = self._mode.get()
        if mode == "Single":
            keys = [("A", "single", "A (single)"), ("B", "single", "B (single)")]
        elif mode == "Ratio":
            keys = [
                ("A", "num", "A (numerator)"),
                ("A", "den", "A (denominator)"),
                ("B", "num", "B (numerator)"),
                ("B", "den", "B (denominator)"),
            ]
        else:  # Difference
            keys = [
                ("A", "left", "A (left)"),
                ("A", "right", "A (right)"),
                ("B", "left", "B (left)"),
                ("B", "right", "B (right)"),
            ]

        valid = [f"{bucket}.{component}" for bucket, component, _label in keys]
        if self._armed not in valid:
            self._armed = valid[0]
        if self._armed_var.get() != self._armed:
            self._armed_var.set(self._armed)

        for bucket, component, label in keys:
            key = f"{bucket}.{component}"
            radio = ttk.Radiobutton(
                self.targets_panel,
                text=label,
                value=key,
                variable=self._armed_var,
                command=lambda k=key: self._set_armed(k),
            )
            radio.pack(side="top", anchor="w", pady=1)
            self._target_radios.append(radio)

        self._update_help()

    def _set_armed(self, key: str) -> None:
        if not key:
            return
        self._armed = key
        if self._armed_var.get() != key:
            self._armed_var.set(key)
        self._update_help()

    def _update_help(self) -> None:
        mode = self._mode.get()
        if mode == "Single":
            tip = "Single mode averages all picks assigned to Selection A or B."
        elif mode == "Ratio":
            tip = (
                "Ratio mode divides the aggregated numerator picks by the denominator picks"
            )
        else:
            tip = (
                "Difference mode subtracts the aggregated right picks from the left picks"
            )
        self._help_label.configure(
            text=f"{tip}\nCurrently armed target: {self._armed.replace('.', ' → ')}"
        )

    def _clear_bucket(self, bucket: str) -> None:
        for component in ("single", "num", "den", "left", "right"):
            self._picks[bucket][component] = {}
        self._refresh_tables_and_emit()

    def _on_autopop(self) -> None:
        if not callable(self._cb_autopop):
            return

        try:
            row = int(self.row_var.get())
            col = int(self.col_var.get())
        except (TypeError, ValueError):
            messagebox.showwarning(
                "Auto-populate", "Row and Column must be integer positions."
            )
            return

        scope = self.scope_var.get() if hasattr(self, "scope_var") else "Selected"
        if scope not in {"Selected", "All"}:
            scope = "Selected"

        self._cb_autopop(self._armed, row, col, scope)

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

        self._refresh_tables_and_emit()

    # ------------------------------------------------------------------ table data
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

    def _refresh_preview(self, tree: ttk.Treeview, dataframe: pd.DataFrame) -> None:
        for item_id in tree.get_children():
            tree.delete(item_id)

        if dataframe is None or dataframe.empty:
            return

        for _, row in dataframe.iterrows():
            file_id = str(row.get("file", ""))
            value = row.get("value", math.nan)
            display = "" if pd.isna(value) else f"{float(value):.6g}"
            tree.insert("", "end", values=(file_id, display))

    def _refresh_tables_and_emit(self) -> None:
        self._populate_pick_tree(self.tree_a, "A")
        self._populate_pick_tree(self.tree_b, "B")

        a_df = self._compute_bucket("A")
        b_df = self._compute_bucket("B")
        self._refresh_preview(self.preview_a, a_df)
        self._refresh_preview(self.preview_b, b_df)

        if callable(self._cb_metrics):
            try:
                self._cb_metrics("Selection A", a_df, "Selection B", b_df)
            except Exception:
                # UI callbacks should never crash the panel; log elsewhere if needed
                pass


__all__ = ["DataTable", "ScrollFrame", "SelectionPanel"]
