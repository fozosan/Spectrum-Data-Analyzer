"""Plot controls and chart rendering for the Tkinter UI."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from math import isfinite

from raman_analyzer.analysis.grouping import compute_error_table
from raman_analyzer.analysis.trendlines import (
    eval_linear,
    eval_power,
    eval_quadratic,
    fit_linear,
    fit_power,
    fit_quadratic,
)
from raman_analyzer.tkui.plot_controller import PlotController


INVERSE_X_LABEL = "Inverse Solve X"

def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


class PlotPanel(ttk.Frame):
    """Encapsulates plot configuration widgets and the Matplotlib canvas."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        session,
        controls_parent: tk.Misc,
    ) -> None:
        super().__init__(master)
        self.session = session
        self._pending_annotations: list[Tuple[np.ndarray, np.ndarray, str]] = []
        self._last_group_stats: Optional[pd.DataFrame] = None
        self._tag_lookup: Optional[Callable[[list[str]], list[str]]] = None
        self._inverse_x_provider: Optional[Callable[[pd.DataFrame, str], Optional[pd.Series]]] = None
        self._buttons: dict[str, ttk.Button] = {}

        # --------------------------- controls ---------------------------
        self.controls_container = ttk.Frame(controls_parent)
        self.controls_container.pack(side="top", fill="x", expand=True)

        # -------------------- Literature Solve (Inverse) --------------------
        inv_box = ttk.LabelFrame(self.controls_container, text="Literature Solve (Inverse)")
        inv_box.pack(side="top", fill="x", padx=6, pady=(6, 6))

        self.inv_model = tk.StringVar(value="Linear")
        ttk.Label(inv_box, text="Model").grid(row=0, column=0, sticky="w")
        self.inv_combo = ttk.Combobox(
            inv_box,
            textvariable=self.inv_model,
            state="readonly",
            width=16,
            values=("Linear", "Quadratic", "Power"),
        )
        self.inv_combo.current(0)
        self.inv_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self.inv_example = ttk.Label(inv_box, text="y = m·x + b", foreground="#555555")
        self.inv_example.grid(row=0, column=2, sticky="w", padx=4)

        self.inv_params_frame = ttk.Frame(inv_box)
        self.inv_params_frame.grid(row=1, column=0, columnspan=6, sticky="ew", pady=(4, 2))
        self.inv_param_vars: dict[str, tk.StringVar] = {}
        # Remember previous model's values when switching; then rebuild UI for new model
        self._last_inv_model = self.inv_model.get()
        self.inv_combo.bind("<<ComboboxSelected>>", self._on_inverse_model_changed)
        # Initialize params from session memory (if any)
        saved_model = getattr(self.session, "inv_model_last", None)
        if saved_model in ("Linear", "Quadratic", "Power"):
            self.inv_model.set(saved_model)
        self._last_inv_model = self.inv_model.get()
        self._refresh_inverse_params()

        ttk.Label(inv_box, text="Y metric").grid(row=2, column=0, sticky="w")
        self.inv_y_metric = tk.StringVar(value="")
        self.inv_y_combo = ttk.Combobox(inv_box, textvariable=self.inv_y_metric, state="readonly", width=24)
        self.inv_y_combo.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(inv_box, text="Source").grid(row=2, column=2, sticky="w")
        self.inv_y_source = tk.StringVar(value="Points")
        self.inv_source_combo = ttk.Combobox(
            inv_box,
            textvariable=self.inv_y_source,
            state="readonly",
            width=12,
            values=("Points", "Group mean"),
        )
        self.inv_source_combo.current(0)
        self.inv_source_combo.grid(row=2, column=3, sticky="w", padx=4, pady=2)

        self.inv_plot_on_chart = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            inv_box,
            text="Plot solutions on chart",
            variable=self.inv_plot_on_chart,
        ).grid(row=2, column=4, sticky="w", padx=6)
        ttk.Button(inv_box, text="Solve", command=self._on_inverse_solve).grid(
            row=2, column=5, padx=6, pady=2, sticky="w"
        )

        # Restore last-used source/plot selections if available
        saved_source = getattr(self.session, "inv_y_source_last", None)
        if saved_source in ("Points", "Group mean"):
            self.inv_y_source.set(saved_source)
            try:
                self.inv_source_combo.set(saved_source)
            except Exception:
                pass
        saved_plot_flag = getattr(self.session, "inv_plot_on_chart_last", None)
        if isinstance(saved_plot_flag, bool):
            self.inv_plot_on_chart.set(bool(saved_plot_flag))

        self.inverse_table = ttk.Treeview(inv_box, columns=("label", "y", "x1", "x2"), show="headings", height=6)
        for column, width in (("label", 200), ("y", 110), ("x1", 120), ("x2", 120)):
            self.inverse_table.heading(column, text=column.upper())
            anchor = "center" if column != "label" else "w"
            self.inverse_table.column(column, width=width, anchor=anchor)
        self.inverse_table.grid(row=3, column=0, columnspan=6, sticky="ew", padx=2, pady=(2, 6))

        inverse_export_row = ttk.Frame(inv_box)
        inverse_export_row.grid(row=4, column=0, columnspan=6, sticky="w", pady=(0, 4))
        ttk.Button(inverse_export_row, text="Export Solutions", command=self._export_inverse).pack(side="left", padx=2)
        ttk.Button(inverse_export_row, text="Copy Solutions", command=self._copy_inverse).pack(side="left", padx=2)

        control_box = ttk.LabelFrame(self.controls_container, text="Plot")
        control_box.pack(side="top", fill="x", padx=6, pady=6)

        self.x_field = tk.StringVar(value="Ordering")
        self.y_field = tk.StringVar(value="")
        self.group_field = tk.StringVar(value="Tag")
        self.plot_type = tk.StringVar(value="Scatter")
        # Distribution / error visualization mode. "Error bars" toggle kept for legacy state storage.
        self.dist_mode = tk.StringVar(value="None")
        self.show_err = tk.BooleanVar(value=False)
        self.x_label_text = tk.StringVar(value="")
        self.y_label_text = tk.StringVar(value="")
        self.auto_x = tk.BooleanVar(value=True)
        self.auto_y = tk.BooleanVar(value=True)
        # Where to adjust defaults later: tweak auto range defaults, entry widths, or combo selections.
        self._derived_metrics: tuple[str, ...] = ()
        self._last_series_for_xticks: list[tuple[float, str]] = []

        ttk.Label(control_box, text="X").grid(row=0, column=0, sticky="w")
        self.x_combo = ttk.Combobox(control_box, textvariable=self.x_field, state="readonly", width=18)
        self.x_combo["values"] = ("Ordering", "Tag (numeric)")
        self.x_combo.current(0)
        self.x_combo.grid(row=0, column=1, padx=4, pady=2, sticky="w")

        ttk.Label(control_box, text="Y").grid(row=0, column=2, sticky="w")
        self.y_combo = ttk.Combobox(control_box, textvariable=self.y_field, state="readonly", width=30)
        self.y_combo.grid(row=0, column=3, padx=4, pady=2, sticky="w")

        ttk.Label(control_box, text="Group").grid(row=0, column=4, sticky="w")
        self.group_combo = ttk.Combobox(control_box, textvariable=self.group_field, state="readonly", width=14)
        self.group_combo["values"] = ("Tag", "None")
        self.group_combo.current(0)
        self.group_combo.grid(row=0, column=5, padx=4, pady=2, sticky="w")

        ttk.Label(control_box, text="Type").grid(row=0, column=6, sticky="w")
        self.type_combo = ttk.Combobox(control_box, textvariable=self.plot_type, state="readonly", width=10)
        self.type_combo["values"] = ("Scatter", "Line")
        self.type_combo.current(0)
        self.type_combo.grid(row=0, column=7, padx=4, pady=2, sticky="w")

        ttk.Label(control_box, text="Dist/Errors").grid(row=0, column=8, sticky="w")
        self.dist_combo = ttk.Combobox(control_box, textvariable=self.dist_mode, state="readonly", width=12)
        self.dist_combo["values"] = ("None", "Mean±SEM", "Mean±Std", "95% CI", "Box", "Violin")
        self.dist_combo.current(0)
        self.dist_combo.grid(row=0, column=9, padx=4, pady=2, sticky="w")

        ttk.Button(control_box, text="Plot", command=self._on_plot).grid(row=0, column=10, padx=6, pady=2, sticky="e")
        clear_btn = ttk.Button(control_box, text="Clear Plot", command=self._on_clear_plot)
        clear_btn.grid(row=0, column=11, padx=6, pady=2, sticky="w")
        self._buttons["clear_plot"] = clear_btn

        self.use_tag_text_for_ordering = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            control_box,
            text="Use Tag (text) instead of Ordering",
            variable=self.use_tag_text_for_ordering,
            command=self._on_toggle_tag_text_for_ordering,
        ).grid(row=1, column=0, columnspan=11, sticky="w", padx=4, pady=(6, 2))

        label_row = ttk.Frame(control_box)
        label_row.grid(row=2, column=0, columnspan=11, sticky="ew", pady=(4, 0))
        ttk.Label(label_row, text="X label").pack(side="left")
        self.x_label_entry = ttk.Entry(label_row, textvariable=self.x_label_text, width=20)
        self.x_label_entry.pack(side="left", padx=(4, 12))
        ttk.Label(label_row, text="Y label").pack(side="left")
        self.y_label_entry = ttk.Entry(label_row, textvariable=self.y_label_text, width=20)
        self.y_label_entry.pack(side="left", padx=(4, 0))

        range_row = ttk.Frame(control_box)
        range_row.grid(row=3, column=0, columnspan=11, sticky="ew", pady=(4, 0))
        ttk.Checkbutton(
            range_row,
            text="Auto X range",
            variable=self.auto_x,
            command=self._update_range_state,
        ).pack(
            side="left", padx=(0, 6)
        )
        ttk.Label(range_row, text="X min").pack(side="left")
        self.x_min_entry = ttk.Entry(range_row, width=10)
        self.x_min_entry.pack(side="left", padx=(4, 6))
        ttk.Label(range_row, text="X max").pack(side="left")
        self.x_max_entry = ttk.Entry(range_row, width=10)
        self.x_max_entry.pack(side="left", padx=(4, 12))
        ttk.Checkbutton(
            range_row,
            text="Auto Y range",
            variable=self.auto_y,
            command=self._update_range_state,
        ).pack(
            side="left", padx=(0, 6)
        )
        ttk.Label(range_row, text="Y min").pack(side="left")
        self.y_min_entry = ttk.Entry(range_row, width=10)
        self.y_min_entry.pack(side="left", padx=(4, 6))
        ttk.Label(range_row, text="Y max").pack(side="left")
        self.y_max_entry = ttk.Entry(range_row, width=10)
        self.y_max_entry.pack(side="left")

        export_row = ttk.Frame(control_box)
        export_row.grid(row=4, column=0, columnspan=11, sticky="w", pady=(4, 0))
        ttk.Button(export_row, text="Export XY", command=self._export_xy).pack(side="left", padx=2)
        ttk.Button(export_row, text="Copy XY", command=self._copy_xy).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Plot (PNG)", command=self._export_plot).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Group Stats", command=self._export_group_stats).pack(side="left", padx=2)
        ttk.Button(export_row, text="Copy Stats", command=self._copy_group_stats).pack(side="left", padx=2)

        self._update_range_state()

        # -------------------- fit (Trendline only) --------------------
        fit_box = ttk.LabelFrame(self.controls_container, text="Trendline")
        fit_box.pack(side="top", fill="x", padx=6, pady=(0, 6))

        self.fit_model = tk.StringVar(value="Linear")
        ttk.Label(fit_box, text="Model").grid(row=0, column=0, sticky="w")
        self.fit_combo = ttk.Combobox(
            fit_box,
            textvariable=self.fit_model,
            state="readonly",
            width=16,
            values=("Linear", "Quadratic", "Power"),
        )
        self.fit_combo.current(0)
        self.fit_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Button(fit_box, text="Fit", command=self._on_fit).grid(row=0, column=2, padx=6, pady=2, sticky="w")
        ttk.Button(fit_box, text="Clear fit", command=self._clear_fit).grid(
            row=0, column=3, padx=4, pady=2, sticky="w"
        )

        self.fit_summary = tk.Text(fit_box, height=4, width=60)
        self.fit_summary.grid(row=1, column=0, columnspan=5, sticky="ew", padx=2, pady=2)
        self.fit_summary.configure(state="disabled")

        residual_buttons = ttk.Frame(fit_box)
        residual_buttons.grid(row=2, column=0, columnspan=5, sticky="w", padx=2, pady=(2, 2))
        ttk.Button(residual_buttons, text="Export Residuals", command=self._export_residuals).pack(
            side="left", padx=2
        )
        ttk.Button(residual_buttons, text="Copy Residuals", command=self._copy_residuals).pack(
            side="left", padx=2
        )

        # --------------------------- figure ---------------------------
        self.figure = Figure(figsize=(6, 4), dpi=120)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.plot_controller = PlotController(self.canvas, self.axes)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=6, pady=(0, 6))
        # Expose full plot controls (pan/zoom/home/save)
        # NavigationToolbar2Tk auto-packs itself when constructed.
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        # Keep toolbar alive across clears; never destroy self.axes/self.figure.

        # keep an internal buffer for inverse annotations until a plot is drawn
        # (buffer already initialized in __init__ above)

        # --------------------------- state ---------------------------
        self._current_xy = pd.DataFrame(columns=["file", "tag", "x", "y", "__group__"])
        self._fit: dict[str, object] | None = None
        self._fit_fn = None
        self._fit_label: str | None = None

    def _refresh_inverse_params(self) -> None:
        # Rebuild the inverse-parameter inputs based on the current model,
        # restoring last-used params for this model from session memory.
        previous = {name: var.get() for name, var in getattr(self, "inv_param_vars", {}).items()}
        # Clear the frame
        for child in self.inv_params_frame.winfo_children():
            child.destroy()
        # Reset storage
        self.inv_param_vars = {}

        model = self.inv_model.get()
        # Pull stored params for this model (session-level, flat)
        stored = {}
        if hasattr(self.session, "get_inv_params"):
            try:
                stored = dict(self.session.get_inv_params(model))
            except Exception:
                stored = {}
        # Compose defaults preferring stored->previous->hard defaults
        defaults = {
            "Linear": {
                "m": stored.get("m", previous.get("m", "1.0")),
                "b": stored.get("b", previous.get("b", "0.0")),
            },
            "Quadratic": {
                "a": stored.get("a", previous.get("a", "1.0")),
                "b": stored.get("b", previous.get("b", "0.0")),
                "c": stored.get("c", previous.get("c", "0.0")),
            },
            "Power": {
                "a": stored.get("a", previous.get("a", "1.0")),
                "b": stored.get("b", previous.get("b", "1.0")),
            },
        }
        examples = {
            "Linear": "y = m·x + b",
            "Quadratic": "y = a·x² + b·x + c",
            "Power": "y = a·xᵇ (a>0)",
        }
        # Update example text if present
        if hasattr(self, "inv_example"):
            self.inv_example.configure(text=examples.get(model, ""))

        params = defaults.get(model, {})
        row = 0
        for name, default_value in params.items():
            ttk.Label(self.inv_params_frame, text=name).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=default_value)
            entry = ttk.Entry(self.inv_params_frame, textvariable=var, width=12)
            entry.grid(row=row, column=1, sticky="w", padx=(6, 12), pady=2)
            self.inv_param_vars[name] = var
            row += 1

        for col in range(2):
            self.inv_params_frame.columnconfigure(col, weight=0)

    def _save_inverse_params_for(self, model: str) -> None:
        if not model or not hasattr(self.session, "set_inv_params"):
            return
        params = {}
        for name, var in getattr(self, "inv_param_vars", {}).items():
            params[name] = var.get()
        try:
            self.session.set_inv_params(model, params)
        except Exception:
            pass

    def _on_inverse_model_changed(self, *_):
        # Save params of the *previous* model before switching UI
        prev = getattr(self, "_last_inv_model", None)
        if prev:
            self._save_inverse_params_for(prev)
        # Update "last" and persist model choice
        curr = self.inv_model.get()
        self._last_inv_model = curr
        if hasattr(self.session, "inv_model_last"):
            self.session.inv_model_last = curr
        # Rebuild inputs with stored params for the new model
        self._refresh_inverse_params()

    def _on_inverse_solve(self) -> None:
        for item_id in self.inverse_table.get_children():
            self.inverse_table.delete(item_id)

        try:
            params = {name: float(var.get()) for name, var in self.inv_param_vars.items()}
        except (TypeError, ValueError):
            messagebox.showwarning("Inverse", "All parameters must be numeric.")
            return
        # Persist current model & params
        if hasattr(self.session, "set_inv_params"):
            try:
                self.session.set_inv_params(self.inv_model.get(), params)
                self.session.inv_model_last = self.inv_model.get()
            except Exception:
                pass

        inverse_fn = self._inverse_for(self.inv_model.get(), params)
        if inverse_fn is None:
            messagebox.showwarning("Inverse", "Unsupported model or invalid parameters.")
            return

        df = getattr(self.session, "results_df", None)
        if df is None or df.empty:
            messagebox.showinfo("Inverse", "No results available. Compute selections first.")
            return

        y_metric = (self.inv_y_metric.get() or "").strip()
        if not y_metric:
            messagebox.showwarning("Inverse", "Choose a Y metric.")
            return
        # Persist last-used Y metric/source/plot flag
        try:
            self.session.inv_y_metric_last = y_metric
            if hasattr(self, "inv_y_source"):
                self.session.inv_y_source_last = self.inv_y_source.get()
            if hasattr(self, "inv_plot_on_chart"):
                self.session.inv_plot_on_chart_last = bool(self.inv_plot_on_chart.get())
        except Exception:
            pass

        work = df.copy()
        work["y"] = self._resolve_axis(y_metric, work)
        work = work[[col for col in ("file", "tag", "y") if col in work.columns]]
        work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])
        if work.empty:
            messagebox.showinfo("Inverse", f"No finite values for {y_metric}.")
            return

        results: list[tuple[str, float, float, float]] = []
        if self.inv_y_source.get() == "Group mean":
            tmp = work.copy()
            if "tag" in tmp.columns:
                tmp["__group__"] = tmp["tag"].astype(str)
            else:
                tmp["__group__"] = "All"
            means = tmp.groupby("__group__")["y"].mean().reset_index()
            for _, row in means.iterrows():
                y_val = float(row["y"])
                label = str(row["__group__"]) or "All"
                sols = [float(val) for val in inverse_fn(y_val) if isfinite(val)]
                x1 = sols[0] if len(sols) >= 1 else float("nan")
                x2 = sols[1] if len(sols) >= 2 else float("nan")
                results.append((label, y_val, x1, x2))
        else:
            for _, row in work.iterrows():
                y_val = float(row["y"])
                label = str(row.get("tag") or row.get("file") or "") or "(unnamed)"
                sols = [float(val) for val in inverse_fn(y_val) if isfinite(val)]
                x1 = sols[0] if len(sols) >= 1 else float("nan")
                x2 = sols[1] if len(sols) >= 2 else float("nan")
                results.append((label, y_val, x1, x2))

        if not results:
            self.inverse_table.insert("", "end", values=("—", "—", "—", "—"))
            return

        xs_to_plot: list[float] = []
        ys_to_plot: list[float] = []
        for label, y_val, x1, x2 in results:
            display = (
                label,
                f"{y_val:.6g}",
                "" if not isfinite(x1) else f"{x1:.6g}",
                "" if not isfinite(x2) else f"{x2:.6g}",
            )
            self.inverse_table.insert("", "end", values=display)
            for candidate in (x1, x2):
                if isfinite(candidate):
                    xs_to_plot.append(candidate)
                    ys_to_plot.append(y_val)

        if self.inv_plot_on_chart.get() and xs_to_plot:
            self.add_annotation_points(
                np.asarray(xs_to_plot, dtype=float),
                np.asarray(ys_to_plot, dtype=float),
                label="Inverse solutions",
            )

        # --- Persist inverse solutions into results_df as a plottable metric ---
        try:
            df_res = getattr(self.session, "results_df", None)
            if df_res is None or df_res.empty or "file" not in df_res.columns:
                # Nothing to persist against
                pass
            else:
                # Helper: choose the first finite candidate among [x1, x2]
                def _choose_x(x1: float, x2: float) -> float:
                    if isfinite(x1):
                        return float(x1)
                    if isfinite(x2):
                        return float(x2)
                    return float("nan")

                y_metric = (self.inv_y_metric.get() or "").strip()
                if not y_metric:
                    # Already validated above; guard anyway
                    return

                # Rebuild the same work frame used for solving so we can map per-file.
                work = df_res.copy()
                work["y"] = self._resolve_axis(y_metric, work)
                cols = [c for c in ("file", "tag", "y") if c in work.columns]
                work = work.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])

                source = (self.inv_y_source.get() or "Points").strip()
                file_to_values: dict[str, list[float]] = {}

                if source == "Group mean":
                    # Map each group label to its solved X, then replicate to all files with that tag.
                    tag_map: dict[str, float] = {}
                    for label, y_val, x1, x2 in results:
                        label_str = str(label)
                        x_val = _choose_x(x1, x2)
                        if isfinite(x_val) and label_str:
                            tag_map[label_str] = x_val

                    if "tag" in df_res.columns:
                        for _, row in df_res[["file", "tag"]].iterrows():
                            f = str(row["file"])
                            t = "" if pd.isna(row["tag"]) else str(row["tag"])
                            x_val = tag_map.get(t, float("nan"))
                            if isfinite(x_val):
                                file_to_values.setdefault(f, []).append(x_val)
                else:
                    # Points: compute per-row solutions and aggregate per file (mean of finite values).
                    for _, row in work.iterrows():
                        f = str(row.get("file", ""))
                        yv = float(row["y"])
                        sols = [float(val) for val in inverse_fn(yv) if isfinite(val)]
                        x1 = sols[0] if len(sols) >= 1 else float("nan")
                        x2 = sols[1] if len(sols) >= 2 else float("nan")
                        xv = _choose_x(x1, x2)
                        if f and isfinite(xv):
                            file_to_values.setdefault(f, []).append(xv)

                if file_to_values:
                    files, values = [], []
                    for f, arr in file_to_values.items():
                        if arr:
                            files.append(f)
                            values.append(float(np.nanmean(np.asarray(arr, dtype=float))))
                    if files:
                        persist_df = pd.DataFrame({"file": files, "value": values})
                        self.session.update_metric(INVERSE_X_LABEL, persist_df)
                        # refresh X/Y menus to include the new metric
                        latest = [c for c in self.session.results_df.columns if c not in {"file", "tag"}]
                        self.set_metrics_for_xy(latest)
        except Exception:
            # Do not interrupt UI flow if persistence fails.
            pass

    # ------------------------------------------------------------------ public API
    def set_tag_lookup(self, fn: Callable[[list[str]], list[str]]) -> None:
        self._tag_lookup = fn

    def set_inverse_x_provider(
        self, fn: Callable[[pd.DataFrame, str], Optional[pd.Series]]
    ) -> None:
        # Retained for compatibility, but plotting no longer calls providers; import-only policy.
        self._inverse_x_provider = fn

    def _metric_choices_base(self, names: Sequence[str]) -> Tuple[str, ...]:
        metrics = [str(name) for name in names if name]
        metric_set = set(metrics)

        derived: list[str] = []
        if {"Selection A", "Selection B"} <= metric_set:
            derived = [
                "Selection A / Selection B",
                "Selection B / Selection A",
                "Selection A - Selection B",
                "Selection B - Selection A",
            ]

        self._derived_metrics = tuple(derived)

        choices: list[str] = ["Ordering", "Tag (numeric)"]
        choices.extend(metrics)
        choices.extend(derived)
        choices.append(INVERSE_X_LABEL)
        return tuple(choices)

    def set_metrics_for_xy(self, names: Sequence[str]) -> None:
        values = self._metric_choices_base(names)
        inv_values = tuple(v for v in values if v != INVERSE_X_LABEL)

        self.x_combo["values"] = values
        self.y_combo["values"] = values
        self.inv_y_combo["values"] = inv_values

        # Strict: clear selections that are no longer available (no fallbacks).
        # X and Y validate against full 'values'; inverse-Y validates against 'inv_values'.
        for var, combo, pool in (
            (self.x_field, self.x_combo, values),
            (self.y_field, self.y_combo, values),
            (self.inv_y_metric, self.inv_y_combo, inv_values),
        ):
            current = var.get()
            if current not in pool:
                var.set("")
                combo.set("")
        # Restore last-used inverse Y metric if empty and available
        last_inv_y = getattr(self.session, "inv_y_metric_last", None)
        if (self.inv_y_metric.get() == "") and last_inv_y and (last_inv_y in inv_values):
            self.inv_y_metric.set(last_inv_y)
            try:
                self.inv_y_combo.set(last_inv_y)
            except Exception:
                pass

    def _resolve_inverse_x_series(self, df: pd.DataFrame) -> pd.Series:
        # Import-only policy: require a precomputed column named exactly INVERSE_X_LABEL.
        if INVERSE_X_LABEL not in df.columns:
            messagebox.showwarning(
                "Plot",
                f"{INVERSE_X_LABEL} is not available. Compute and save it first.",
            )
            raise RuntimeError("Inverse Solve X column missing.")
        return pd.to_numeric(df[INVERSE_X_LABEL], errors="coerce")

    def set_metrics(self, names: Sequence[str]) -> None:
        self.set_metrics_for_xy(names)

    def _on_toggle_tag_text_for_ordering(self) -> None:
        # No automatic plotting; users must replot explicitly after toggling.
        return

    def _on_clear_plot(self) -> None:
        # Do NOT clear the figure; preserve toolbar bindings.
        try:
            self.axes.cla()
        except Exception:
            # If axes were ever missing, recreate them once (defensive; not expected)
            self.axes = self.figure.add_subplot(111)
            self.plot_controller.axes = self.axes
        # Reset internal buffers but keep fit only if user clears it explicitly.
        self._current_xy = pd.DataFrame(columns=["file", "tag", "x", "y", "__group__"])
        self._last_group_stats = None
        self.axes.set_xlabel(self.x_label_text.get().strip() or "")
        self.axes.set_ylabel(self.y_label_text.get().strip() or "")
        self.canvas.draw_idle()

    def _update_range_state(self) -> None:
        state_x = "disabled" if self.auto_x.get() else "normal"
        state_y = "disabled" if self.auto_y.get() else "normal"
        for widget in (self.x_min_entry, self.x_max_entry):
            widget.configure(state=state_x)
        for widget in (self.y_min_entry, self.y_max_entry):
            widget.configure(state=state_y)

    def _coerce_x_if_use_tag_text(
        self, df: pd.DataFrame, x_choice: str, x_series: pd.Series
    ) -> pd.Series:
        if not getattr(self, "use_tag_text_for_ordering", None):
            return x_series
        if not self.use_tag_text_for_ordering.get():
            return x_series
        if str(x_choice) != "Ordering":
            return x_series

        if "file" not in df.columns:
            messagebox.showwarning(
                "Plot",
                "Cannot use Tag (text) for X because 'file' column is missing.",
            )
            raise RuntimeError("Missing 'file' column for tag mapping.")

        if self._tag_lookup is None:
            messagebox.showwarning(
                "Plot",
                "Cannot use Tag (text) for X because tag lookup is not available.",
            )
            raise RuntimeError("Tag lookup callback not set.")

        file_ids = [str(f) for f in df["file"].tolist()]
        tags = self._tag_lookup(file_ids)
        if len(tags) != len(file_ids):
            messagebox.showwarning(
                "Plot",
                "Tag lookup did not return the expected number of entries.",
            )
            raise RuntimeError("Tag lookup length mismatch.")
        clean_tags = ["" if t is None else str(t).strip() for t in tags]
        if any(tag == "" for tag in clean_tags):
            messagebox.showwarning(
                "Plot",
                "Tag is missing for one or more rows; cannot replace Ordering with Tag (text).",
            )
            raise RuntimeError("Missing tag values.")

        df["__tag_text_for_ordering__"] = clean_tags
        df["__x_numeric__"] = pd.to_numeric(x_series, errors="coerce")
        df["tag"] = clean_tags
        return pd.Series(clean_tags, index=df.index)

    # ------------------------------------------------------------------ plotting helpers
    def _build_xy(self, x_label: str, y_label: str) -> pd.DataFrame | None:
        df = self.session.results_df
        if df is None or df.empty:
            return None

        work = df.copy()
        try:
            x_source = self._resolve_axis(x_label, work)
        except RuntimeError:
            return None
        try:
            coerced = self._coerce_x_if_use_tag_text(work, x_label, x_source)
        except RuntimeError:
            return None
        work["x"] = coerced
        if "__x_numeric__" not in work.columns:
            work["__x_numeric__"] = pd.to_numeric(x_source, errors="coerce")
        try:
            work["y"] = self._resolve_axis(y_label, work)
        except RuntimeError:
            return None
        work = work.replace([np.inf, -np.inf], np.nan)
        cols = ["file", "tag", "x", "y", "__x_numeric__", "__tag_text_for_ordering__"]
        existing_cols = [c for c in cols if c in work.columns]
        work = work[existing_cols]
        work = work.dropna(subset=["x", "y"])
        if "__x_numeric__" in work.columns:
            work = work.dropna(subset=["__x_numeric__"])
        if work.empty:
            return None

        group_mode = self.group_field.get()
        if group_mode == "Tag" and "tag" in work.columns:
            work["__group__"] = work["tag"].astype(str)
        else:
            work["__group__"] = "All"
        return work

    def _resolve_axis(self, label: str, work: pd.DataFrame) -> pd.Series:
        if str(label) == INVERSE_X_LABEL:
            return self._resolve_inverse_x_series(work)
        if label == "Ordering":
            mapping = dict(getattr(self.session, "ordering", {}) or {})
            files = work["file"] if "file" in work.columns else pd.Series(index=work.index, dtype=object)
            return files.map(lambda fid: _safe_float(mapping.get(str(fid))))
        if label == "Tag (numeric)":
            tags = work["tag"] if "tag" in work.columns else pd.Series(index=work.index, dtype=object)
            return tags.map(_safe_float)
        if label in self._derived_metrics:
            return self._derive_series(work, label)

        series = work.get(label)
        if series is None:
            return pd.Series(np.nan, index=work.index)
        return pd.to_numeric(series, errors="coerce")

    def _derive_series(self, work: pd.DataFrame, label: str) -> pd.Series:
        if not {"Selection A", "Selection B"} <= set(work.columns):
            return pd.Series(np.nan, index=work.index)

        a = pd.to_numeric(work.get("Selection A"), errors="coerce")
        b = pd.to_numeric(work.get("Selection B"), errors="coerce")

        with np.errstate(divide="ignore", invalid="ignore"):
            if label == "Selection A / Selection B":
                result = a / b
            elif label == "Selection B / Selection A":
                result = b / a
            elif label == "Selection A - Selection B":
                result = a - b
            elif label == "Selection B - Selection A":
                result = b - a
            else:
                return pd.Series(np.nan, index=work.index)

        if isinstance(result, pd.Series):
            return result.replace([np.inf, -np.inf], np.nan)
        return pd.Series(np.nan, index=work.index)

    def _parse_range(self, min_value: str, max_value: str, axis_name: str) -> tuple[float, float]:
        min_text = (min_value or "").strip()
        max_text = (max_value or "").strip()
        if not min_text or not max_text:
            raise ValueError(f"Provide both minimum and maximum for the {axis_name} range.")
        try:
            vmin = float(min_text)
            vmax = float(max_text)
        except ValueError as exc:
            raise ValueError(f"{axis_name} range must be numeric.") from exc
        if vmin == vmax:
            raise ValueError(f"{axis_name} range minimum and maximum must differ.")
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        return (vmin, vmax)

    def _on_plot(self) -> None:
        x_label = self.x_field.get().strip()
        y_label = self.y_field.get().strip()
        if not y_label:
            messagebox.showinfo("Plot", "Choose a Y metric to plot.")
            return
        if not x_label:
            messagebox.showinfo("Plot", "Choose an X source to plot.")
            return

        df = self.session.results_df
        if df is None or df.empty:
            messagebox.showinfo("Plot", "No data available to plot.")
            return

        def _axis_available(label: str) -> bool:
            if label == INVERSE_X_LABEL:
                if INVERSE_X_LABEL in df.columns:
                    return True
                messagebox.showwarning(
                    "Plot",
                    f"{INVERSE_X_LABEL} is not available. Compute and save it first.",
                )
                return False
            if label in {"Ordering", "Tag (numeric)"}:
                return True
            if label in self._derived_metrics:
                return {"Selection A", "Selection B"} <= set(df.columns)
            return label in df.columns

        if not _axis_available(x_label):
            if x_label != INVERSE_X_LABEL:
                messagebox.showinfo("Plot", f"No data available for X: {x_label}.")
            return

        if not _axis_available(y_label):
            if y_label != INVERSE_X_LABEL:
                messagebox.showinfo("Plot", f"No data available for Y: {y_label}.")
            return

        work = self._build_xy(x_label, y_label)
        if work is None or work.empty:
            messagebox.showinfo("Plot", "No valid X/Y pairs to plot.")
            return

        self._current_xy = work.copy()
        self._last_group_stats = None

        x_axis_label = self.x_label_text.get().strip() or x_label
        y_axis_label = self.y_label_text.get().strip() or y_label
        self._last_series_for_xticks = []
        if x_label == "Ordering":
            self._last_series_for_xticks = self._collect_ordering_ticks(work)

        try:
            x_limits = None if self.auto_x.get() else self._parse_range(self.x_min_entry.get(), self.x_max_entry.get(), "X")
        except ValueError as exc:
            messagebox.showerror("Plot", str(exc))
            return

        try:
            y_limits = None if self.auto_y.get() else self._parse_range(self.y_min_entry.get(), self.y_max_entry.get(), "Y")
        except ValueError as exc:
            messagebox.showerror("Plot", str(exc))
            return

        series_entries: list[dict[str, object]] = []
        numeric_col = "__x_numeric__" if "__x_numeric__" in work.columns else None
        for label, group_df in work.groupby("__group__"):
            source = group_df[numeric_col] if numeric_col else group_df["x"]
            xs = pd.to_numeric(source, errors="coerce").to_numpy(dtype=float)
            ys = pd.to_numeric(group_df["y"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(xs) & np.isfinite(ys)
            xs = xs[mask]
            ys = ys[mask]
            if xs.size == 0 or ys.size == 0:
                continue
            legend_label = None if label in ("", "All") else label
            series_entries.append({"x": xs, "y": ys, "label": legend_label})

        dist_mode = (self.dist_mode.get() or "None").strip()
        valid_modes = {"None", "Mean±SEM", "Mean±Std", "95% CI", "Box", "Violin"}
        if dist_mode not in valid_modes:
            messagebox.showwarning("Plot", f"Unsupported distribution mode: {dist_mode}.")
            dist_mode = "None"
            self.dist_mode.set(dist_mode)

        error_entries: list[dict[str, object]] = []
        if dist_mode in {"Mean±SEM", "Mean±Std", "95% CI"}:
            grouped = self._compute_group_stats(work, mode=dist_mode)
            if grouped is None or grouped.empty:
                messagebox.showinfo("Plot", f"No grouped statistics available for {dist_mode}.")
            else:
                for grp_label, grp_df in grouped.groupby("__group__"):
                    if "__x_numeric__" in grp_df.columns:
                        xs_source = grp_df["__x_numeric__"]
                    else:
                        xs_source = grp_df["x"]
                    xs = pd.to_numeric(xs_source, errors="coerce").to_numpy(dtype=float)
                    means = grp_df["mean"].to_numpy(dtype=float)
                    yerr = grp_df.get("yerr")
                    err = None if yerr is None else yerr.to_numpy(dtype=float)
                    legend = None if grp_label in ("", "All") else f"{grp_label} ({dist_mode})"
                    error_entries.append({"x": xs, "mean": means, "yerr": err, "label": legend})

        self.plot_controller.clear_fit()

        self.plot_controller.draw_scatter(
            series_entries,
            x_label=x_axis_label,
            y_label=y_axis_label,
            xlim=x_limits,
            ylim=y_limits,
            plot_type=self.plot_type.get(),
            error_series=error_entries if error_entries else None,
        )

        self._flush_pending_annotations()

        if dist_mode in {"Box", "Violin"}:
            try:
                self._draw_box_violin(work, mode=dist_mode)
            except Exception as exc:
                messagebox.showwarning(dist_mode, f"{dist_mode} draw failed: {exc}")

        if x_label == "Ordering" and self._last_series_for_xticks:
            try:
                xs, names = zip(*self._last_series_for_xticks)
                self.axes.set_xticks(xs)
                self.axes.set_xticklabels(names, rotation=45, ha="right")
            except Exception:
                messagebox.showwarning("Plot", "Failed to apply Ordering tick labels.")

        current_xlim = self.axes.get_xlim()

        if self._fit is not None and self._fit_fn is not None and not work.empty:
            try:
                xs = pd.to_numeric(work["x"], errors="coerce").to_numpy(dtype=float)
                xs = xs[np.isfinite(xs)]
                if xs.size == 0:
                    raise ValueError("No finite X values for fit")
                label = self._fit_label or f"Fit: {self._fit['model']}"
                self.plot_controller.draw_fit(self._fit_fn, x_data=xs, label=label)
                current_xlim = self.axes.get_xlim()
            except Exception:
                pass

        handles, labels = self.axes.get_legend_handles_labels()
        if labels:
            self.axes.legend(loc="best")

        # Reapply manual limits in case overlays nudged the autoscale.
        if x_limits is not None:
            self.axes.set_xlim(x_limits)
        if y_limits is not None:
            self.axes.set_ylim(y_limits)

        self.canvas.draw_idle()

    # -------------------------- distribution helpers --------------------------
    def _compute_group_stats(self, work: pd.DataFrame, *, mode: str) -> pd.DataFrame | None:
        if work is None or work.empty:
            return None

        df = work.copy()
        if "__group__" not in df.columns:
            df["__group__"] = "All"

        numeric_map: pd.DataFrame | None = None
        if "__x_numeric__" in df.columns:
            numeric_map = (
                df[["x", "__x_numeric__"]]
                .dropna()
                .drop_duplicates()
            )

        grouped = (
            df.groupby(["__group__", "x"])["y"]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        if numeric_map is not None and not numeric_map.empty:
            grouped = grouped.merge(numeric_map, on="x", how="left")
        if grouped.empty:
            self._last_group_stats = pd.DataFrame()
            return grouped

        counts = grouped["count"].to_numpy(dtype=float)
        std = grouped["std"].to_numpy(dtype=float)
        std = np.where(np.isfinite(std), std, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sem = np.divide(std, np.sqrt(np.maximum(counts, 1.0)))
        sem = np.where(np.isfinite(sem), sem, 0.0)
        ci95 = 1.96 * sem

        result = grouped.copy()
        if "__x_numeric__" in result.columns:
            result["__x_numeric__"] = pd.to_numeric(result["__x_numeric__"], errors="coerce")
        result["std"] = std
        result["sem"] = sem
        result["ci95"] = ci95
        if mode == "Mean±Std":
            result["yerr"] = std
        elif mode == "95% CI":
            result["yerr"] = ci95
        else:
            result["yerr"] = sem
        result["mode"] = mode
        self._last_group_stats = result.copy()
        return result[["__group__", "x", "mean", "yerr"]]

    def _draw_box_violin(self, work: pd.DataFrame, *, mode: str) -> None:
        if work is None or work.empty:
            raise ValueError("No data available for distribution plot.")

        df = work.copy()
        if "__group__" not in df.columns:
            df["__group__"] = "All"

        if "__x_numeric__" in df.columns:
            df["__x_numeric__"] = pd.to_numeric(df["__x_numeric__"], errors="coerce")
            df = df.dropna(subset=["__x_numeric__"])
            x_numeric_col = "__x_numeric__"
        else:
            df["x"] = pd.to_numeric(df["x"], errors="coerce")
            x_numeric_col = "x"
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
        if df.empty:
            raise ValueError("No finite values for distribution plot.")

        stats = df[["__group__", x_numeric_col, "y"]].copy()
        stats = stats.rename(columns={x_numeric_col: "x"})
        stats["mode"] = mode
        self._last_group_stats = stats

        ax = self.axes
        groups = sorted(df["__group__"].astype(str).unique())
        x_values = df[x_numeric_col].dropna().unique()
        if x_values.size == 0:
            raise ValueError("No finite X values for distribution plot.")
        xs_sorted = sorted(float(v) for v in x_values)

        if len(groups) > 1:
            offsets = np.linspace(-0.25, 0.25, len(groups))
        else:
            offsets = np.array([0.0])

        width = 0.35 if len(groups) > 1 else 0.5

        any_drawn = False
        for idx, group in enumerate(groups):
            subset = df[df["__group__"] == group]
            data_per_x = []
            positions = []
            for x_val in xs_sorted:
                ys = subset.loc[np.isclose(subset[x_numeric_col], x_val), "y"]
                ys = ys[np.isfinite(ys)]
                if ys.empty:
                    continue
                data_per_x.append(list(ys.values))
                positions.append(x_val + offsets[idx])

            if not data_per_x:
                continue

            any_drawn = True
            if mode == "Box":
                ax.boxplot(data_per_x, positions=positions, widths=width, manage_ticks=False)
            else:
                ax.violinplot(
                    data_per_x,
                    positions=positions,
                    widths=width,
                    showmeans=True,
                    showextrema=True,
                    showmedians=True,
                )

        if not any_drawn:
            raise ValueError("No finite values for distribution plot.")

    def _collect_ordering_ticks(self, work: pd.DataFrame) -> list[tuple[float, str]]:
        ticks: list[tuple[float, str]] = []
        seen: set[tuple[float, str]] = set()
        numeric_col = "__x_numeric__" if "__x_numeric__" in work.columns else None
        label_col = "x" if numeric_col else "tag"
        for _, row in work.iterrows():
            source_value = row.get(numeric_col) if numeric_col else row.get("x")
            x_val = _safe_float(source_value)
            tag = str(row.get(label_col, ""))
            if not tag:
                continue
            if not isfinite(x_val):
                continue
            key = (x_val, tag)
            if key in seen:
                continue
            seen.add(key)
            ticks.append((x_val, tag))
        return sorted(ticks, key=lambda item: item[0])

    # ------------------------------------------------------------------ exports
    def _export_inverse(self) -> None:
        rows = [self.inverse_table.item(iid)["values"] for iid in self.inverse_table.get_children()]
        if not rows:
            messagebox.showinfo("Export Solutions", "Nothing to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        pd.DataFrame(rows, columns=["label", "y", "x1", "x2"]).to_csv(path, index=False)

    def _copy_inverse(self) -> None:
        rows = [self.inverse_table.item(iid)["values"] for iid in self.inverse_table.get_children()]
        if not rows:
            messagebox.showinfo("Copy", "Nothing to copy.")
            return
        df = pd.DataFrame(rows, columns=["label", "y", "x1", "x2"])
        self._copy_text(df.to_csv(index=False))

    def _export_xy(self) -> None:
        if self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Export XY", "Nothing to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        export_cols: Iterable[str] = [c for c in ("file", "tag", "x", "y") if c in self._current_xy.columns]
        self._current_xy.loc[:, export_cols].to_csv(path, index=False)

    def _export_plot(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        self.figure.savefig(path, bbox_inches="tight", dpi=200)

    def _export_group_stats(self) -> None:
        dist_mode = (self.dist_mode.get() or "None").strip()

        if dist_mode in {"Mean±SEM", "Mean±Std", "95% CI"}:
            if self._current_xy is None or self._current_xy.empty:
                messagebox.showinfo("Export Group Stats", "Plot data before exporting statistics.")
                return
            grouped = self._compute_group_stats(self._current_xy, mode=dist_mode)
            if grouped is None or grouped.empty:
                messagebox.showinfo(
                    "Export Group Stats",
                    f"No grouped statistics available for {dist_mode}.",
                )
                return
            export_df = grouped
        else:
            if self._current_xy is None or self._current_xy.empty:
                messagebox.showinfo("Export Group Stats", "Plot data before exporting statistics.")
                return
            renamed = self._current_xy.rename(columns={"__group__": "tag"})
            export_df = compute_error_table(renamed, mode="SEM")
            if export_df is None or export_df.empty:
                messagebox.showinfo("Export Group Stats", "No grouped statistics available.")
                return
            self._last_group_stats = export_df.copy()

        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        export_df.to_csv(path, index=False)

    def _export_intersections(self) -> None:
        raise RuntimeError("Intersections feature has been removed.")

    def _export_residuals(self) -> None:
        if self._fit is None or self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Residuals", "Fit and plot are required.")
            return
        if self._fit_fn is None:
            messagebox.showinfo("Residuals", "Fit function is not available.")
            return
        try:
            # Residuals must be computed on the same X used for fitting.
            x_raw = self._current_xy["x"]
            x_series = pd.to_numeric(x_raw, errors="coerce")
            if x_series.isna().any():
                messagebox.showerror(
                    "Residuals",
                    "Selected X axis contains non-numeric values; residuals cannot be exported.",
                )
                return
            x_vals = x_series.to_numpy(dtype=float)
            predicted = np.asarray(self._fit_fn(x_vals), dtype=float)
        except Exception:
            messagebox.showinfo("Residuals", "Unable to evaluate fitted model.")
            return

        export = self._current_xy.copy()
        export = export.assign(
            y_fit=predicted,
            residual=export["y"].to_numpy(dtype=float) - predicted,
        )
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        export.to_csv(path, index=False)

    # ------------------------------------------------------------------ fitting & math helpers
    def _clear_fit(self) -> None:
        self._fit = None
        self._fit_fn = None
        self._fit_label = None
        self.plot_controller.clear_fit()
        self.fit_summary.configure(state="normal")
        self.fit_summary.delete("1.0", "end")
        self.fit_summary.configure(state="disabled")
        self.canvas.draw_idle()

    def _on_fit(self) -> None:
        if self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Fit", "Plot some data first.")
            return

        # Fit on exactly the X axis that is currently selected for plotting.
        # If the X axis is non-numeric (e.g., tag text), fail with a clear message.
        x_raw = self._current_xy["x"]
        x_series = pd.to_numeric(x_raw, errors="coerce")
        y_series = pd.to_numeric(self._current_xy["y"], errors="coerce")

        if x_series.isna().any():
            try:
                bad_examples = (
                    x_raw[x_series.isna()]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )[:5]
            except Exception:
                bad_examples = []
            msg = "Selected X axis contains non-numeric values. Fitting requires numeric X."
            if bad_examples:
                msg += f"\nExamples: {', '.join(bad_examples)}"
            messagebox.showerror("Fit", msg)
            return

        x = x_series.to_numpy(dtype=float)
        y = y_series.to_numpy(dtype=float)

        try:
            if self.fit_model.get() == "Linear":
                result = fit_linear(x, y)
                coeffs = result["coeffs"]
                model_name = "Linear"
                m, b = (float(coeffs[0]), float(coeffs[1]))
                fit_fn = lambda t, M=m, B=b: eval_linear(t, M, B)
            elif self.fit_model.get() == "Quadratic":
                result = fit_quadratic(x, y)
                coeffs = result["coeffs"]
                model_name = "Quadratic"
                a, b, c = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
                fit_fn = lambda t, A=a, B=b, C=c: eval_quadratic(t, A, B, C)
            else:
                result = fit_power(x, y)
                coeffs = result["coeffs"]
                model_name = "Power"
                a, b = (float(coeffs[0]), float(coeffs[1]))
                fit_fn = lambda t, A=a, B=b: eval_power(t, A, B)
        except Exception as exc:
            messagebox.showwarning("Fit", f"Unable to compute fit: {exc}")
            return

        self._fit = {
            "model": model_name,
            "coeffs": tuple(float(c) for c in coeffs),
            "r2": float(result.get("r2", np.nan)),
        }
        self._fit_fn = fit_fn
        self._fit_label = f"Fit: {model_name}"

        summary_lines = [
            f"Model: {model_name}",
            f"Coefficients: {self._fit['coeffs']}",
            f"R²: {self._fit['r2']:.4f}" if np.isfinite(self._fit["r2"]) else "R²: n/a",
        ]
        self.fit_summary.configure(state="normal")
        self.fit_summary.delete("1.0", "end")
        self.fit_summary.insert("end", "\n".join(summary_lines) + "\n")
        self.fit_summary.configure(state="disabled")

        self._on_plot()

    def _refresh_literature_params(self) -> None:
        raise RuntimeError("Literature Overlay has been removed.")

    def _on_intersections(self) -> None:
        raise RuntimeError("Intersections feature has been removed.")

    def _overlay_literature(self) -> None:
        raise RuntimeError("Literature Overlay has been removed.")

    def _clear_literature(self) -> None:
        raise RuntimeError("Literature Overlay has been removed.")

    def _inverse_for(self, model: str, params: dict[str, float]):
        if model == "Linear":
            m = float(params.get("m", 0.0))
            b = float(params.get("b", 0.0))
            if m == 0:
                return None

            def inv_linear(y: float) -> list[float]:
                return [(y - b) / m]

            return inv_linear

        if model == "Power":
            a = float(params.get("a", 1.0))
            b = float(params.get("b", 1.0))
            if a <= 0 or b == 0:
                return None

            def inv_power(y: float) -> list[float]:
                ratio = y / a
                if ratio < 0:
                    return []
                try:
                    value = float(np.power(ratio, 1.0 / b))
                except Exception:
                    return []
                return [value]

            return inv_power

        if model == "Quadratic":
            A = float(params.get("a", 0.0))
            B = float(params.get("b", 0.0))
            C = float(params.get("c", 0.0))

            def inv_quadratic(y: float) -> list[float]:
                a = A
                b = B
                c = C - y
                if a == 0:
                    if b == 0:
                        return []
                    return [(-c) / b]
                discriminant = b * b - 4 * a * c
                if discriminant < 0:
                    return []
                root = float(np.sqrt(discriminant))
                return [(-b - root) / (2 * a), (-b + root) / (2 * a)]

            return inv_quadratic

        return None

    # ------------------------------ Annotations -------------------------------
    def add_annotation_points(self, xs: np.ndarray, ys: np.ndarray, label: str = "Points") -> None:
        data = (
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            label,
        )
        # Drop any identical queued annotation (same label and coordinates).
        self._pending_annotations = [
            existing
            for existing in self._pending_annotations
            if not (
                existing[2] == label
                and np.array_equal(existing[0], data[0])
                and np.array_equal(existing[1], data[1])
            )
        ]
        self._pending_annotations.append(data)
        try:
            self.plot_controller.draw_points(data[0], data[1], label=label, style_kwargs={"color": "#1E88E5"})
            self.canvas.draw_idle()
            self._pending_annotations.pop()  # already drawn
        except Exception:
            # keep queued for future plots
            pass

    def _flush_pending_annotations(self) -> None:
        if not self._pending_annotations:
            return
        queued: list[tuple[np.ndarray, np.ndarray, str]] = []
        seen: set[tuple[str, tuple[float, ...], tuple[float, ...]]] = set()
        for xs, ys, label in self._pending_annotations:
            key = (label, tuple(xs.tolist()), tuple(ys.tolist()))
            if key in seen:
                continue
            try:
                self.plot_controller.draw_points(xs, ys, label=label, style_kwargs={"color": "#1E88E5"})
            except Exception:
                queued.append((xs, ys, label))
            seen.add(key)
        self._pending_annotations = queued
        if not queued:
            self.canvas.draw_idle()

    # ------------------------------ Clipboard ---------------------------------
    def _copy_text(self, text: str) -> None:
        try:
            self.clipboard_clear()
            self.clipboard_append(text)
        except Exception as exc:
            messagebox.showwarning("Copy", f"Clipboard error: {exc}")

    def _copy_xy(self) -> None:
        if self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Copy", "Nothing to copy.")
            return
        cols = [col for col in ("file", "tag", "x", "y", "__group__") if col in self._current_xy.columns]
        self._copy_text(self._current_xy.loc[:, cols].to_csv(index=False))

    def _copy_group_stats(self) -> None:
        if self._last_group_stats is None or self._last_group_stats.empty:
            messagebox.showinfo("Copy", "No grouped statistics to copy. Plot with a Dist/Errors mode first.")
            return
        self._copy_text(self._last_group_stats.to_csv(index=False))

    def _copy_intersections(self) -> None:
        raise RuntimeError("Intersections feature has been removed.")

    def _copy_residuals(self) -> None:
        if self._fit_fn is None or self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Copy", "Residuals are not available.")
            return
        # Residuals must reflect the chosen X axis; fail if non-numeric.
        x_raw = self._current_xy["x"]
        x_series = pd.to_numeric(x_raw, errors="coerce")
        if x_series.isna().any():
            messagebox.showinfo(
                "Copy", "Selected X axis contains non-numeric values; residuals unavailable."
            )
            return
        x_vals = x_series.to_numpy(dtype=float)
        try:
            predicted = np.asarray(self._fit_fn(x_vals), dtype=float)
        except Exception:
            messagebox.showinfo("Copy", "Unable to evaluate fitted model.")
            return
        export = self._current_xy.copy()
        export = export.assign(y_fit=predicted, residual=export["y"].to_numpy(dtype=float) - predicted)
        self._copy_text(export.to_csv(index=False))


    def _make_literature_function(
        self, model: str, params: dict[str, float]
    ) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        raise RuntimeError("Literature Overlay has been removed.")


__all__ = ["PlotPanel"]
