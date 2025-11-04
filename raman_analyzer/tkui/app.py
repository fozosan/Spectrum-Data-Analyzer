"""Tkinter front-end for the Raman Analyzer application."""
from __future__ import annotations

import tkinter as tk
from tkinter import (
    filedialog,
    messagebox,
    simpledialog,
    ttk,
)
from typing import Any, Dict, List, Optional

import pandas as pd

from raman_analyzer.models.session import AnalysisSession
from raman_analyzer.tkui.plot_panel import PlotPanel
from raman_analyzer.tkui.widgets import DataTable, FileList, ScrollFrame, SelectionPanel


class TkRamanApp:
    """Main application controller for the Tkinter Raman Analyzer UI."""

    def __init__(self, root: tk.Tk, session: Optional[AnalysisSession] = None):
        self.root = root
        self.root.title("Raman Analyzer (Tk)")
        self.session = session or AnalysisSession()
        self.current_file: Optional[str] = None

        self._ui_built = False
        self._build_menu()
        self._build_ui()

    # ------------------------------------------------------------------ UI setup
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(label="Load CSVs…", command=self._load_csvs)
        menu_file.add_command(label="Import mapping CSV…", command=self._import_mapping_csv)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=menu_file)
        self.root.config(menu=menubar)

    def _build_ui(self) -> None:
        # If UI was already constructed, just retarget session-aware widgets.
        if getattr(self, "_ui_built", False):
            if hasattr(self, "file_list"):
                self.file_list.session = self.session
            if hasattr(self, "selection_panel"):
                self.selection_panel.session = self.session
            if hasattr(self, "plot_panel"):
                self.plot_panel.session = self.session
            self._refresh_plot_metrics()
            return

        main_split = ttk.Panedwindow(self.root, orient="horizontal")
        main_split.pack(fill="both", expand=True)

        # --------------------------- left pane (files + table)
        left_container = ttk.Frame(main_split)
        main_split.add(left_container, weight=1)
        left_split = ttk.Panedwindow(left_container, orient="vertical")
        left_split.pack(fill="both", expand=True)

        files_box = ttk.LabelFrame(left_split, text="Files")
        self.file_list = FileList(
            files_box,
            session=self.session,
            on_tag_changed=self._on_file_tag_changed,
            on_x_changed=self._on_file_x_changed,
            on_selection_changed=self._on_file_selection_changed,
        )
        self.file_list.pack(fill="both", expand=True)
        left_split.add(files_box, weight=1)

        table_box = ttk.LabelFrame(left_split, text="Data")
        self.data_table = DataTable(table_box, on_cell_double_click=self._on_cell_double_click)
        self.data_table.pack(fill="both", expand=True)
        left_split.add(table_box, weight=2)

        # --------------------------- right pane (controls + plot)
        # NOTE: controls live in the scrollable right panel; the canvas sits below (in PlotPanel).
        right_container = ttk.Frame(main_split)
        main_split.add(right_container, weight=2)
        right_split = ttk.Panedwindow(right_container, orient="vertical")
        right_split.pack(fill="both", expand=True)

        controls_scroll = ScrollFrame(right_split)
        right_split.add(controls_scroll, weight=3)
        controls = controls_scroll.inner

        self.selection_panel = SelectionPanel(
            controls,
            session=self.session,
            on_metrics=self._on_metrics_updated,
            on_autopopulate=self._on_autopopulate,
        )
        self.selection_panel.pack(side="top", fill="both", expand=True)

        # Plot panel: controls go into 'controls', canvas is the frame we add below.
        # (No implicit defaults; users must choose metrics explicitly.)
        if not hasattr(self, "plot_panel"):
            self.plot_panel = PlotPanel(
                right_container,
                session=self.session,
                controls_parent=controls,
            )

            # Inject a tag lookup callback so PlotPanel can map file_ids → tag strings.
            def _lookup_tags(file_ids: list[str]) -> list[str]:
                return [self.session.file_to_tag.get(fid, "") for fid in file_ids]

            if hasattr(self.plot_panel, "set_tag_lookup"):
                self.plot_panel.set_tag_lookup(_lookup_tags)

            # Inject an optional inverse-solve provider: df, inv_y_name -> pd.Series (x values)
            # NOTE: Strict wiring — if not available in session, we do not fabricate anything.
            def _inverse_x_provider(df, inv_y_name: str):
                if hasattr(self.session, "inverse_solve_x_series"):
                    return self.session.inverse_solve_x_series(df, inv_y_name)
                return None

            if hasattr(self.plot_panel, "set_inverse_x_provider"):
                self.plot_panel.set_inverse_x_provider(_inverse_x_provider)
        right_split.add(self.plot_panel, weight=4)

        self.root.geometry("1280x800")
        self.root.minsize(900, 600)
        self._refresh_plot_metrics()
        self._ui_built = True

    # ------------------------------------------------------------------ data IO
    def _load_csvs(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select peak CSV files", filetypes=[("CSV files", "*.csv")]
        )
        if not paths:
            return

        existing_files = set(self.session.list_files())
        existing_tables = set(getattr(self.session, "raw_tables", {}).keys())
        combined: List[pd.DataFrame] = []
        tables: Dict[str, pd.DataFrame] = {}
        for path in paths:
            file_key = str(path)
            if file_key in existing_tables:
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # pragma: no cover - interactive warning
                messagebox.showwarning("Load CSV", f"Failed to read {path}\n{exc}")
                continue

            if "file" not in df.columns:
                df = df.copy()
                df["file"] = file_key

            combined.append(df)
            tables[file_key] = df.copy()
            existing_tables.add(file_key)

        if not combined:
            return

        merged = pd.concat(combined, ignore_index=True)
        self.session.set_raw_tables(tables)
        self.session.set_raw_data(merged)

        all_files = self.session.list_files()
        just_loaded_files = [fid for fid in all_files if fid not in existing_files]

        current_files: List[str] = []
        if hasattr(self.file_list, "get_files"):
            current_files = self.file_list.get_files()
        combined_listing = list(current_files)
        for fid in all_files:
            if fid not in combined_listing:
                combined_listing.append(fid)
        self.file_list.set_files(combined_listing)

        # --- Batch tag (optional, asked for any count >=1)
        preset_tag = simpledialog.askstring(
            "Apply tag to imported files (optional)",
            "Enter a tag (alphanumeric) to apply to all imported files.\n"
            "(Leave blank to skip.)",
            parent=self.root,
        )
        if preset_tag:
            for fid in just_loaded_files:
                self.session.set_tag(fid, str(preset_tag).strip())
            # refresh UI
            self.selection_panel.set_context(self.session.file_to_tag)
            self.file_list.refresh()

        # --- Ordering (strict; no fallbacks)
        if len(just_loaded_files) == 1:
            raw_value = simpledialog.askstring(
                "Ordering",
                "Enter an Ordering value (numeric) for the imported file.\n"
                "(Leave blank to skip; no defaults will be applied.)",
                parent=self.root,
            )
            if raw_value:
                try:
                    value = float(raw_value)
                    self.session.update_ordering({just_loaded_files[0]: value})
                    self.file_list.refresh()
                except Exception:
                    messagebox.showwarning("Ordering", "Ordering must be numeric. Nothing applied.")
        else:
            if just_loaded_files:
                raw_list = simpledialog.askstring(
                    "Ordering (multiple files)",
                    "Enter comma-separated numeric Ordering values for each imported file, in the same order they were selected.\n"
                    f"Count must be exactly {len(just_loaded_files)}.\n"
                    "(Leave blank to skip; no defaults will be applied.)",
                    parent=self.root,
                )
                if raw_list:
                    parts = [p.strip() for p in raw_list.split(",")]
                    if len(parts) != len(just_loaded_files):
                        messagebox.showwarning(
                            "Ordering",
                            f"Expected {len(just_loaded_files)} values; got {len(parts)}. Nothing applied.",
                        )
                    else:
                        try:
                            values = [float(p) for p in parts]
                            mapping = {fid: val for fid, val in zip(just_loaded_files, values)}
                            self.session.update_ordering(mapping)
                            self.file_list.refresh()
                        except Exception:
                            messagebox.showwarning("Ordering", "All Ordering values must be numeric. Nothing applied.")

        if just_loaded_files:
            self.file_list.select_file(just_loaded_files[0])
        else:
            self.file_list.refresh()

        self.selection_panel.set_context(self.session.file_to_tag)
        self._refresh_plot_metrics()

    def _import_mapping_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select mapping CSV", filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - interactive warning
            messagebox.showwarning("Import mapping CSV", f"Failed to read {path}\n{exc}")
            return

        file_column = next((c for c in df.columns if c.lower() == "file"), None)
        if not file_column:
            messagebox.showwarning(
                "Import mapping CSV", "CSV must include a 'file' column."
            )
            return

        tag_column = next(
            (c for c in df.columns if c.lower() in ("tag", "sample", "name", "label")),
            None,
        )
        order_column = next(
            (c for c in df.columns if c.lower() in ("ordering", "order")),
            None,
        )

        tags_applied = 0
        ordering_map: Dict[str, float] = {}

        if tag_column:
            for _, row in df.iterrows():
                file_id = str(row.get(file_column, "")).strip()
                if not file_id:
                    continue
                tag_value = str(row.get(tag_column, "")).strip()
                try:
                    self.session.set_tag(file_id, tag_value)
                    if tag_value:
                        tags_applied += 1
                except Exception:
                    continue

        if order_column:
            for _, row in df.iterrows():
                file_id = str(row.get(file_column, "")).strip()
                if not file_id:
                    continue
                raw_value = row.get(order_column)
                if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
                    continue
                try:
                    ordering_map[file_id] = float(raw_value)
                except (TypeError, ValueError):
                    continue

        ordering_warning: Optional[str] = None
        if ordering_map:
            if hasattr(self.session, "update_ordering") and callable(
                getattr(self.session, "update_ordering", None)
            ):
                try:
                    self.session.update_ordering(ordering_map)
                except Exception as exc:
                    ordering_warning = f"update_ordering failed: {exc}"
            else:
                ordering_warning = "session missing required API; run ordering migration."
                if hasattr(self.session, "ordering"):
                    try:
                        setattr(self.session, "ordering", dict(ordering_map))
                    except Exception:
                        pass

        self.selection_panel.set_context(self.session.file_to_tag)
        self.file_list.refresh()
        self._refresh_plot_metrics()

        messagebox.showinfo(
            "Import mapping CSV",
            f"Success: Imported {tags_applied} tags and {len(ordering_map)} ordering values.",
        )
        if ordering_warning:
            messagebox.showwarning("Import mapping CSV", ordering_warning)

    # ------------------------------------------------------------------ callbacks
    def _on_file_selection_changed(self, files: List[str]) -> None:
        if not files:
            self.current_file = None
            self.data_table.set_dataframe(pd.DataFrame())
            self.selection_panel.set_context(self.session.file_to_tag)
            return

        self.current_file = files[0]
        table = self.session.get_raw_table(self.current_file)
        if table is not None:
            self.data_table.set_dataframe(table)
            self.selection_panel.set_context(self.session.file_to_tag)
            return

        raw = self.session.raw_df
        if raw is None or raw.empty or "file" not in raw.columns:
            self.data_table.set_dataframe(pd.DataFrame())
            self.selection_panel.set_context(self.session.file_to_tag)
            return

        subset = raw[raw["file"].astype(str) == str(self.current_file)]
        self.data_table.set_dataframe(subset)
        self.selection_panel.set_context(self.session.file_to_tag)

    def _on_file_tag_changed(self, _file_id: str, _tag: str) -> None:
        self.selection_panel.set_context(self.session.file_to_tag)
        self._refresh_plot_metrics()

    def _on_file_x_changed(self, _file_id: str, _value: Optional[float]) -> None:
        self._refresh_plot_metrics()

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
        self,
        a_name: str,
        selection_a: pd.DataFrame,
        b_name: str,
        selection_b: pd.DataFrame,
    ) -> None:
        try:
            if isinstance(selection_a, pd.DataFrame):
                safe_a = (
                    selection_a[["file", "value"]]
                    if not selection_a.empty
                    else pd.DataFrame(columns=["file", "value"])
                )
                self.session.update_metric(a_name, safe_a)
            if isinstance(selection_b, pd.DataFrame):
                safe_b = (
                    selection_b[["file", "value"]]
                    if not selection_b.empty
                    else pd.DataFrame(columns=["file", "value"])
                )
                self.session.update_metric(b_name, safe_b)
        except Exception:  # pragma: no cover - defensive
            pass
        # Keep plot/inverse metric menus in sync with latest selection outputs.
        self._refresh_plot_metrics()

    def _on_autopopulate(
        self, target_key: str, row1: int, col1: int, scope: str
    ) -> None:
        if scope == "All":
            raw = self.session.raw_df
            if raw is not None and not raw.empty and "file" in raw.columns:
                files = raw["file"].astype(str).dropna().unique().tolist()
            else:
                files = []
        else:
            files = self.file_list.get_selected_files()
            if not files and self.current_file:
                files = [self.current_file]

        if not files:
            messagebox.showinfo(
                "Auto-populate", "No files available for the requested scope."
            )
            return

        added = 0
        failures: list[str] = []
        for file_id in files:
            table = self.session.get_raw_table(file_id)
            if table is None or table.empty:
                failures.append(f"{file_id}: no table")
                continue
            try:
                r_idx = max(0, row1 - 1)
                c_idx = max(0, col1 - 1)
                value = float(table.iloc[r_idx, c_idx])
            except Exception:
                failures.append(f"{file_id}: non-numeric or out-of-bounds")
                continue
            tag = self.session.file_to_tag.get(file_id, "")
            self.selection_panel.add_pick(
                file_id,
                row1,
                col1,
                value,
                target=target_key,
                tag=tag,
            )
            added += 1

        if added == 0:
            messagebox.showwarning(
                "Auto-populate",
                "No values added." + ("\n" + "\n".join(failures) if failures else ""),
            )
            return

        if failures:
            messagebox.showwarning(
                "Auto-populate",
                "Auto-populate completed with issues:\n" + "\n".join(failures),
            )

    # Keep PlotPanel’s X/Y menus current with session metrics/results
    def _refresh_plot_metrics(self) -> None:
        try:
            if hasattr(self.session, "list_metrics") and callable(
                getattr(self.session, "list_metrics", None)
            ):
                names = list(self.session.list_metrics())
            elif hasattr(self.session, "metrics"):
                metrics_obj = getattr(self.session, "metrics", {}) or {}
                names = list(metrics_obj.keys())
            else:
                df = getattr(self.session, "results_df", None)
                if df is None:
                    names = []
                else:
                    names = [c for c in df.columns if c not in {"file", "tag"}]
        except Exception as exc:
            messagebox.showwarning("Plot options", f"Could not refresh metrics: {exc}")
            names = []
        if hasattr(self, "plot_panel"):
            self.plot_panel.set_metrics_for_xy(names)


def main() -> None:
    root = tk.Tk()
    TkRamanApp(root)
    root.mainloop()


__all__ = ["TkRamanApp", "main"]
