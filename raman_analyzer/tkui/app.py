"""Tkinter front-end for the Raman Analyzer application."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from raman_analyzer.models.session import AnalysisSession
from raman_analyzer.tkui.widgets import DataTable, ScrollFrame, SelectionPanel


class TkRamanApp:
    """Main application controller for the Tkinter Raman Analyzer UI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Raman Analyzer (Tk)")
        self.session = AnalysisSession()
        self.current_file: Optional[str] = None

        self._build_menu()
        self._build_ui()

    # ------------------------------------------------------------------ UI setup
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(label="Load CSVs…", command=self._load_csvs)
        menu_file.add_command(label="Import Tags/X CSV…", command=self._import_mapping_csv)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=menu_file)
        self.root.config(menu=menubar)

    def _build_ui(self) -> None:
        main_split = ttk.Panedwindow(self.root, orient="horizontal")
        main_split.pack(fill="both", expand=True)

        # --------------------------- left pane (files + table)
        left_container = ttk.Frame(main_split)
        main_split.add(left_container, weight=1)
        left_split = ttk.Panedwindow(left_container, orient="vertical")
        left_split.pack(fill="both", expand=True)

        files_box = ttk.LabelFrame(left_split, text="Files")
        self.files_list = tk.Listbox(files_box, exportselection=False)
        self.files_list.pack(fill="both", expand=True)
        self.files_list.bind("<<ListboxSelect>>", self._on_file_selected)
        left_split.add(files_box, weight=1)

        table_box = ttk.LabelFrame(left_split, text="Data")
        self.data_table = DataTable(table_box, on_cell_double_click=self._on_cell_double_click)
        self.data_table.pack(fill="both", expand=True)
        left_split.add(table_box, weight=2)

        # --------------------------- right pane (controls + plot)
        right_container = ttk.Frame(main_split)
        main_split.add(right_container, weight=2)
        right_split = ttk.Panedwindow(right_container, orient="vertical")
        right_split.pack(fill="both", expand=True)

        controls_scroll = ScrollFrame(right_split)
        right_split.add(controls_scroll, weight=3)
        controls = controls_scroll.inner

        self.selection_panel = SelectionPanel(
            controls, on_metrics_updated=self._on_metrics_updated
        )
        self.selection_panel.pack(side="top", fill="both", expand=True)

        plot_box = ttk.LabelFrame(controls, text="Plot")
        plot_box.pack(side="top", fill="x", padx=6, pady=6)
        ttk.Label(plot_box, text="Y metric").pack(side="top", anchor="w")
        self.y_metric = tk.StringVar(value="Selection A")
        ttk.Combobox(
            plot_box,
            textvariable=self.y_metric,
            values=["Selection A", "Selection B"],
            state="readonly",
        ).pack(side="top", fill="x")
        ttk.Button(plot_box, text="Plot", command=self._plot).pack(
            side="top", pady=(6, 0), anchor="w"
        )

        chart_box = ttk.LabelFrame(right_split, text="Chart")
        right_split.add(chart_box, weight=2)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_box)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, chart_box)

        self.root.geometry("1280x800")
        self.root.minsize(900, 600)

    # ------------------------------------------------------------------ data IO
    def _load_csvs(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select peak CSV files", filetypes=[("CSV files", "*.csv")]
        )
        if not paths:
            return

        combined: List[pd.DataFrame] = []
        tables: Dict[str, pd.DataFrame] = {}
        for path in paths:
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - interactive warning
                messagebox.showwarning("Load CSV", f"Failed to read {path}\n{exc}")
                continue

            if "file" not in df.columns:
                df = df.copy()
                df["file"] = str(path)

            combined.append(df)
            tables[str(path)] = df.copy()

        if not combined:
            return

        merged = pd.concat(combined, ignore_index=True)
        self.session.set_raw_data(merged)
        self.session.set_raw_tables(tables)

        files = (
            merged["file"].astype(str).dropna().unique().tolist()
            if "file" in merged.columns
            else []
        )
        self.files_list.delete(0, "end")
        for file_id in files:
            self.files_list.insert("end", file_id)

        if files:
            self.files_list.select_set(0)
            self._on_file_selected(None)

        self.selection_panel.set_context(self.session.file_to_tag)

    def _import_mapping_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Tags/X mapping CSV", filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - interactive warning
            messagebox.showwarning("Import Tags/X CSV", f"Failed to read {path}\n{exc}")
            return

        file_column = next((c for c in df.columns if c.lower() == "file"), None)
        if not file_column:
            messagebox.showwarning(
                "Import Tags/X CSV", "CSV must include a 'file' column."
            )
            return

        tag_column = next((c for c in df.columns if c.lower() == "tag"), None)
        x_column = next(
            (c for c in df.columns if c.lower() in {"x", "x_value", "xvalue"}),
            None,
        )

        tags_applied = 0
        mapping: Dict[str, float] = {}

        for _, row in df.iterrows():
            file_id = str(row.get(file_column, "")).strip()
            if not file_id:
                continue

            if tag_column is not None:
                tag = str(row.get(tag_column, "")).strip()
                self.session.set_tag(file_id, tag)
                tags_applied += 1

            if x_column is not None:
                try:
                    mapping[file_id] = float(row.get(x_column))
                except (TypeError, ValueError):
                    continue

        if mapping:
            self.session.update_x_mapping(mapping)

        self.selection_panel.set_context(self.session.file_to_tag)
        messagebox.showinfo(
            "Import Tags/X CSV",
            f"Imported {tags_applied} tags and {len(mapping)} X values.",
        )

    # ------------------------------------------------------------------ callbacks
    def _on_file_selected(self, _event: Optional[tk.Event]) -> None:
        selection = self.files_list.curselection()
        if not selection:
            self.current_file = None
            self.data_table.set_dataframe(pd.DataFrame())
            return

        self.current_file = self.files_list.get(selection[0])
        table = self.session.get_raw_table(self.current_file)
        if table is not None:
            self.data_table.set_dataframe(table)
            return

        raw = self.session.raw_df
        if raw is None or raw.empty or "file" not in raw.columns:
            self.data_table.set_dataframe(pd.DataFrame())
            return

        subset = raw[raw["file"].astype(str) == str(self.current_file)]
        self.data_table.set_dataframe(subset)

    def _on_cell_double_click(self, row1: int, col1: int, value: Any) -> None:
        if not self.current_file:
            return

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            messagebox.showwarning(
                "Selection", "Selected cell does not contain a numeric value."
            )
            return

        tag = self.session.file_to_tag.get(self.current_file, "")
        self.selection_panel.add_pick(
            self.current_file, row1, col1, numeric_value, tag=tag
        )

    def _on_metrics_updated(
        self, selection_a: pd.DataFrame, selection_b: pd.DataFrame
    ) -> None:
        try:
            if isinstance(selection_a, pd.DataFrame):
                safe_a = (
                    selection_a[["file", "value"]]
                    if not selection_a.empty
                    else pd.DataFrame(columns=["file", "value"])
                )
                self.session.update_metric("Selection A", safe_a)
            if isinstance(selection_b, pd.DataFrame):
                safe_b = (
                    selection_b[["file", "value"]]
                    if not selection_b.empty
                    else pd.DataFrame(columns=["file", "value"])
                )
                self.session.update_metric("Selection B", safe_b)
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------ plotting
    def _plot(self) -> None:
        results = self.session.results_df
        if results is None or results.empty:
            messagebox.showwarning("Plot", "No data available to plot.")
            return

        target = self.y_metric.get()
        if target not in results.columns:
            messagebox.showwarning("Plot", f"No data for {target}. Add picks first.")
            return

        raw = self.session.raw_df
        files: List[str]
        if raw is not None and not raw.empty and "file" in raw.columns:
            files = raw["file"].astype(str).dropna().unique().tolist()
        else:
            files = results["file"].astype(str).dropna().unique().tolist()

        if not files:
            messagebox.showwarning("Plot", "No files loaded to plot.")
            return

        mapping = dict(self.session.x_mapping or {})
        if mapping:
            x_values = [mapping.get(file_id, np.nan) for file_id in files]
        else:
            x_values = list(range(len(files)))

        y_lookup = dict(zip(results["file"].astype(str), results[target]))
        y_values = [y_lookup.get(file_id, np.nan) for file_id in files]

        self.axes.clear()
        self.axes.set_xlabel("X (mapping or index)")
        self.axes.set_ylabel(target)
        self.axes.plot(x_values, y_values, "o-")
        self.axes.grid(True, linestyle="--", alpha=0.4)
        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    TkRamanApp(root)
    root.mainloop()


__all__ = ["TkRamanApp", "main"]
