"""Plot controls and chart rendering for the Tkinter UI."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        self.inv_combo.bind("<<ComboboxSelected>>", lambda *_: self._refresh_inverse_params())
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

        label_row = ttk.Frame(control_box)
        label_row.grid(row=1, column=0, columnspan=11, sticky="ew", pady=(4, 0))
        ttk.Label(label_row, text="X label").pack(side="left")
        self.x_label_entry = ttk.Entry(label_row, textvariable=self.x_label_text, width=20)
        self.x_label_entry.pack(side="left", padx=(4, 12))
        ttk.Label(label_row, text="Y label").pack(side="left")
        self.y_label_entry = ttk.Entry(label_row, textvariable=self.y_label_text, width=20)
        self.y_label_entry.pack(side="left", padx=(4, 0))

        range_row = ttk.Frame(control_box)
        range_row.grid(row=2, column=0, columnspan=11, sticky="ew", pady=(4, 0))
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
        export_row.grid(row=3, column=0, columnspan=11, sticky="w", pady=(4, 0))
        ttk.Button(export_row, text="Export XY", command=self._export_xy).pack(side="left", padx=2)
        ttk.Button(export_row, text="Copy XY", command=self._copy_xy).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Plot (PNG)", command=self._export_plot).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Group Stats", command=self._export_group_stats).pack(side="left", padx=2)
        ttk.Button(export_row, text="Copy Stats", command=self._copy_group_stats).pack(side="left", padx=2)

        self._update_range_state()

        # -------------------- fit / intersections --------------------
        fit_box = ttk.LabelFrame(self.controls_container, text="Trendline & Intersections")
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

        self.intersections_box = tk.Listbox(fit_box, height=5)
        self.intersections_box.grid(row=2, column=0, columnspan=5, sticky="ew", padx=2, pady=(2, 6))
        ttk.Button(fit_box, text="Export Intersections", command=self._export_intersections).grid(
            row=3, column=0, padx=2, pady=2, sticky="w"
        )
        ttk.Button(fit_box, text="Export Residuals", command=self._export_residuals).grid(
            row=3, column=1, padx=2, pady=2, sticky="w"
        )
        ttk.Button(fit_box, text="Copy Intersections", command=self._copy_intersections).grid(
            row=3, column=2, padx=2, pady=2, sticky="w"
        )
        ttk.Button(fit_box, text="Copy Residuals", command=self._copy_residuals).grid(
            row=3, column=3, padx=2, pady=2, sticky="w"
        )

        literature_box = ttk.LabelFrame(fit_box, text="Literature Overlay")
        literature_box.grid(row=4, column=0, columnspan=5, sticky="ew", padx=2, pady=(6, 2))
        self.lit_model = tk.StringVar(value="Linear")
        ttk.Label(literature_box, text="Model").grid(row=0, column=0, sticky="w")
        self.lit_combo = ttk.Combobox(
            literature_box,
            textvariable=self.lit_model,
            state="readonly",
            width=16,
            values=("Linear", "Quadratic", "Power"),
        )
        self.lit_combo.current(0)
        self.lit_combo.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        self.lit_example = ttk.Label(literature_box, text="y = m·x + b", foreground="#555555")
        self.lit_example.grid(row=0, column=2, sticky="w", padx=4)
        self.lit_params_frame = ttk.Frame(literature_box)
        self.lit_params_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 2))
        self.lit_param_vars: dict[str, tk.StringVar] = {}
        self.lit_combo.bind("<<ComboboxSelected>>", lambda *_: self._refresh_literature_params())
        self._refresh_literature_params()
        literature_buttons = ttk.Frame(literature_box)
        literature_buttons.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 2))
        ttk.Button(literature_buttons, text="Add curve", command=self._overlay_literature).pack(side="left", padx=2)
        ttk.Button(literature_buttons, text="Clear", command=self._clear_literature).pack(side="left", padx=2)
        ttk.Button(literature_buttons, text="Intersections", command=self._on_intersections).pack(side="left", padx=2)
        for column in range(3):
            literature_box.columnconfigure(column, weight=1)

        # --------------------------- figure ---------------------------
        self.figure = Figure(figsize=(6, 4), dpi=120)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.plot_controller = PlotController(self.canvas, self.axes)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=6, pady=(0, 6))

        # keep an internal buffer for inverse annotations until a plot is drawn
        # (buffer already initialized in __init__ above)

        # --------------------------- state ---------------------------
        self._current_xy = pd.DataFrame(columns=["file", "tag", "x", "y", "__group__"])
        self._fit: dict[str, object] | None = None
        self._fit_fn = None
        self._fit_label: str | None = None
        self._literature: list[dict[str, object]] = []

    # ------------------------------------------------------------------ public API
    def set_metrics_for_xy(self, names: Sequence[str]) -> None:
        metrics = [str(name) for name in names if name]
        metric_set = {m for m in metrics}

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

        values = tuple(choices)
        self.x_combo["values"] = values
        self.y_combo["values"] = values
        self.inv_y_combo["values"] = values

        def _ensure_selection(var: tk.StringVar, combo: ttk.Combobox, fallbacks: Sequence[str]) -> None:
            current = var.get()
            if current in values:
                combo.current(values.index(current))
                return
            for candidate in fallbacks:
                if candidate in values:
                    var.set(candidate)
                    combo.current(values.index(candidate))
                    return
            var.set("")
            combo.set("")

        x_fallbacks = [
            self.x_field.get(),
            "Ordering",
            "Tag (numeric)",
            *metrics,
            *derived,
        ]
        _ensure_selection(self.x_field, self.x_combo, x_fallbacks)
        x_choice = self.x_field.get()

        preferred_y = [name for name in ("Selection A", "Selection B", *derived) if name in values]
        y_fallbacks = [
            self.y_field.get(),
            *preferred_y,
            *[item for item in values if item not in {"Ordering", "Tag (numeric)", x_choice}],
            *[item for item in ("Ordering", "Tag (numeric)") if item != x_choice],
        ]
        _ensure_selection(self.y_field, self.y_combo, y_fallbacks)

        inv_preferred = [name for name in ("Selection A", "Selection B", *derived) if name in values]
        inv_fallbacks = [
            self.inv_y_metric.get(),
            *inv_preferred,
            *[item for item in values if item not in {"Ordering", "Tag (numeric)"}],
            *[item for item in ("Ordering", "Tag (numeric)")],
        ]
        _ensure_selection(self.inv_y_metric, self.inv_y_combo, inv_fallbacks)

    def set_metrics(self, names: Sequence[str]) -> None:
        self.set_metrics_for_xy(names)

    def _update_range_state(self) -> None:
        state_x = "disabled" if self.auto_x.get() else "normal"
        state_y = "disabled" if self.auto_y.get() else "normal"
        for widget in (self.x_min_entry, self.x_max_entry):
            widget.configure(state=state_x)
        for widget in (self.y_min_entry, self.y_max_entry):
            widget.configure(state=state_y)

    # ------------------------------------------------------------------ plotting helpers
    def _build_xy(self, x_label: str, y_label: str) -> pd.DataFrame | None:
        df = self.session.results_df
        if df is None or df.empty:
            return None

        work = df.copy()
        work["x"] = self._resolve_axis(x_label, work)
        work["y"] = self._resolve_axis(y_label, work)
        work = work.replace([np.inf, -np.inf], np.nan)
        cols = ["file", "tag", "x", "y"]
        existing_cols = [c for c in cols if c in work.columns]
        work = work[existing_cols]
        work = work.dropna(subset=["x", "y"])
        if work.empty:
            return None

        group_mode = self.group_field.get()
        if group_mode == "Tag" and "tag" in work.columns:
            work["__group__"] = work["tag"].astype(str)
        else:
            work["__group__"] = "All"
        return work

    def _resolve_axis(self, label: str, work: pd.DataFrame) -> pd.Series:
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
            if label in {"Ordering", "Tag (numeric)"}:
                return True
            if label in self._derived_metrics:
                return {"Selection A", "Selection B"} <= set(df.columns)
            return label in df.columns

        if not _axis_available(x_label):
            messagebox.showinfo("Plot", f"No data available for X: {x_label}.")
            return

        if not _axis_available(y_label):
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
        for label, group_df in work.groupby("__group__"):
            xs = group_df["x"].to_numpy(dtype=float)
            ys = group_df["y"].to_numpy(dtype=float)
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
                    xs = grp_df["x"].to_numpy(dtype=float)
                    means = grp_df["mean"].to_numpy(dtype=float)
                    yerr = grp_df.get("yerr")
                    err = None if yerr is None else yerr.to_numpy(dtype=float)
                    legend = None if grp_label in ("", "All") else f"{grp_label} ({dist_mode})"
                    error_entries.append({"x": xs, "mean": means, "yerr": err, "label": legend})

        self.plot_controller.clear_fit()
        self.plot_controller.clear_literature()
        self.plot_controller.clear_crosses()

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
                xs = work["x"].to_numpy(dtype=float)
                xs = xs[np.isfinite(xs)]
                if xs.size == 0:
                    raise ValueError("No finite X values for fit")
                label = self._fit_label or f"Fit: {self._fit['model']}"
                self.plot_controller.draw_fit(self._fit_fn, x_data=xs, label=label)
                current_xlim = self.axes.get_xlim()
            except Exception:
                pass

        for entry in self._literature:
            fn = entry.get("fn")
            label = entry.get("label") or f"Lit: {entry.get('model', '')}"
            if fn is None:
                continue
            try:
                xmin, xmax = current_xlim
                if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin == xmax:
                    continue
                x_grid = np.linspace(xmin, xmax, 256)
                with np.errstate(all="ignore"):
                    y_grid = fn(x_grid)
                mask = np.isfinite(x_grid) & np.isfinite(y_grid)
                if not np.any(mask):
                    continue
                self.plot_controller.draw_literature(
                    x_grid[mask],
                    y_grid[mask],
                    label=label,
                    fn=fn,
                )
            except Exception:
                continue

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

        grouped = (
            df.groupby(["__group__", "x"])["y"]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        if grouped.empty:
            return grouped

        counts = grouped["count"].to_numpy(dtype=float)
        std = grouped["std"].to_numpy(dtype=float)
        sem = std / np.sqrt(np.maximum(counts, 1.0))

        if mode == "Mean±SEM":
            yerr = sem
        elif mode == "Mean±Std":
            yerr = std
        else:  # 95% CI
            yerr = 1.96 * sem

        grouped = grouped.rename(columns={"mean": "mean"})
        grouped["yerr"] = yerr
        result = grouped[["__group__", "x", "mean", "yerr"]]
        self._last_group_stats = result.copy()
        return result

    def _draw_box_violin(self, work: pd.DataFrame, *, mode: str) -> None:
        if work is None or work.empty:
            raise ValueError("No data available for distribution plot.")

        df = work.copy()
        if "__group__" not in df.columns:
            df["__group__"] = "All"

        ax = self.axes
        groups = sorted(df["__group__"].astype(str).unique())
        x_values = pd.to_numeric(df["x"], errors="coerce").dropna().unique()
        if x_values.size == 0:
            raise ValueError("No finite X values for distribution plot.")
        xs_sorted = sorted(float(v) for v in x_values)

        if len(groups) > 1:
            offsets = np.linspace(-0.25, 0.25, len(groups))
        else:
            offsets = np.array([0.0])

        width = 0.35 if len(groups) > 1 else 0.5

        for idx, group in enumerate(groups):
            subset = df[df["__group__"] == group]
            data_per_x = []
            positions = []
            for x_val in xs_sorted:
                ys = pd.to_numeric(
                    subset.loc[np.isclose(subset["x"], x_val), "y"], errors="coerce"
                )
                ys = ys[np.isfinite(ys)]
                data_per_x.append(list(ys.values))
                positions.append(x_val + offsets[idx])

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

    def _collect_ordering_ticks(self, work: pd.DataFrame) -> list[tuple[float, str]]:
        ticks: list[tuple[float, str]] = []
        seen: set[tuple[float, str]] = set()
        for _, row in work.iterrows():
            x_val = _safe_float(row.get("x"))
            tag = str(row.get("tag", ""))
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
        if self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Export Group Stats", "Nothing to export.")
            return
        mode = (self.dist_mode.get() or "None").strip()
        if mode in {"Mean±SEM", "Mean±Std", "95% CI"}:
            grouped = self._compute_group_stats(self._current_xy, mode=mode)
            if grouped is None:
                grouped = pd.DataFrame()
        else:
            renamed = self._current_xy.rename(columns={"__group__": "tag"})
            grouped = compute_error_table(renamed, mode="SEM")
        if grouped.empty:
            messagebox.showinfo("Export Group Stats", "No grouped statistics available.")
            return
        self._last_group_stats = grouped.copy()
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        grouped.to_csv(path, index=False)

    def _export_intersections(self) -> None:
        if self.intersections_box.size() == 0:
            messagebox.showinfo("Export Intersections", "Nothing to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        rows = [self.intersections_box.get(idx) for idx in range(self.intersections_box.size())]
        pd.DataFrame({"point": rows}).to_csv(path, index=False)

    def _export_residuals(self) -> None:
        if self._fit is None or self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Residuals", "Fit and plot are required.")
            return
        if self._fit_fn is None:
            messagebox.showinfo("Residuals", "Fit function is not available.")
            return
        try:
            x_vals = self._current_xy["x"].to_numpy(dtype=float)
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
        self.intersections_box.delete(0, tk.END)
        self.canvas.draw_idle()

    def _on_fit(self) -> None:
        if self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Fit", "Plot some data first.")
            return

        x = self._current_xy["x"].to_numpy(dtype=float)
        y = self._current_xy["y"].to_numpy(dtype=float)

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
        previous = {name: var.get() for name, var in self.lit_param_vars.items()}
        for widget in self.lit_params_frame.winfo_children():
            widget.destroy()
        self.lit_param_vars.clear()

        model = self.lit_model.get()
        defaults: dict[str, dict[str, str]] = {
            "Linear": {"m": previous.get("m", "1.0"), "b": previous.get("b", "0.0")},
            "Quadratic": {
                "a": previous.get("a", "1.0"),
                "b": previous.get("b", "0.0"),
                "c": previous.get("c", "0.0"),
            },
            "Power": {"a": previous.get("a", "1.0"), "b": previous.get("b", "1.0")},
        }
        examples = {
            "Linear": "y = m·x + b",
            "Quadratic": "y = a·x² + b·x + c",
            "Power": "y = a·xᵇ (a>0)",
        }
        self.lit_example.configure(text=examples.get(model, ""))

        params = defaults.get(model, {})
        for row_index, (name, default_value) in enumerate(params.items()):
            ttk.Label(self.lit_params_frame, text=name).grid(row=row_index, column=0, sticky="w")
            var = tk.StringVar(value=default_value)
            entry = ttk.Entry(self.lit_params_frame, textvariable=var, width=12)
            entry.grid(row=row_index, column=1, sticky="w", padx=(6, 12), pady=2)
            self.lit_param_vars[name] = var

        for column in range(2):
            self.lit_params_frame.columnconfigure(column, weight=0)

    def _on_intersections(self) -> None:
        self.intersections_box.delete(0, tk.END)
        if self._fit_fn is None:
            messagebox.showinfo(
                "Intersections",
                "No trendline has been fitted yet.\n\nFit a trendline to the current data before finding intersections.",
            )
            self.intersections_box.insert(tk.END, "Fit required first.")
            return
        if not self._literature:
            self.intersections_box.insert(tk.END, "No literature overlays.")
            self.plot_controller.clear_crosses()
            return

        xmin, xmax = self.axes.get_xlim()
        if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin == xmax:
            if self._current_xy is not None and not self._current_xy.empty:
                xs = self._current_xy["x"].to_numpy(dtype=float)
                xs = xs[np.isfinite(xs)]
                if xs.size:
                    xmin = float(np.nanmin(xs))
                    xmax = float(np.nanmax(xs))
        if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin == xmax:
            self.intersections_box.insert(tk.END, "Invalid plot range.")
            self.plot_controller.clear_crosses()
            return

        sample_x = np.linspace(xmin, xmax, 512)
        with np.errstate(all="ignore"):
            fit_y = np.asarray(self._fit_fn(sample_x), dtype=float)
        mask_fit = np.isfinite(fit_y)
        if not np.any(mask_fit):
            self.intersections_box.insert(tk.END, "Fit is undefined in range.")
            self.plot_controller.clear_crosses()
            return

        sample_x = sample_x[mask_fit]
        fit_y = fit_y[mask_fit]

        seen: set[tuple[int, int]] = set()
        hits_x: list[float] = []
        hits_y: list[float] = []
        found = False

        # Gather analytic sources from both the stored metadata and rendered lines.
        source_functions: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
        fn_ids: set[int] = set()

        for line in getattr(self.plot_controller, "_literature_lines", []):
            fn_line = getattr(line, "_literature_fn", None)
            if callable(fn_line):
                label_line = getattr(line, "_literature_label", None) or "Literature"
                fn_ids.add(id(fn_line))
                source_functions.append((label_line, fn_line))

        for entry in self._literature:
            fn = entry.get("fn")
            if not callable(fn) or id(fn) in fn_ids:
                continue
            label = entry.get("label", "Literature")
            fn_ids.add(id(fn))
            source_functions.append((label, fn))

        for curve_label, fn in source_functions:
            with np.errstate(all="ignore"):
                lit_y = np.asarray(fn(sample_x), dtype=float)
            if lit_y.shape != sample_x.shape:
                continue
            mask = np.isfinite(lit_y)
            if not np.any(mask):
                continue
            xs = sample_x[mask]
            fit_vals = fit_y[mask]
            lit_vals = lit_y[mask]
            if xs.size < 2:
                continue
            diff = fit_vals - lit_vals

            zero_indices = np.where(np.isclose(diff, 0.0, atol=1e-9))[0]
            for idx in zero_indices:
                xi = float(xs[idx])
                yi_arr = self._fit_fn(np.asarray([xi]))
                yi = float(yi_arr[0] if np.ndim(yi_arr) else yi_arr)
                key = (round(xi, 9), round(yi, 9))
                if key in seen:
                    continue
                seen.add(key)
                hits_x.append(xi)
                hits_y.append(yi)
                self.intersections_box.insert(
                    tk.END, f"{curve_label}: x={xi:.6g}, y={yi:.6g}"
                )
                found = True

            signs = np.sign(diff)
            sign_changes = np.where(np.diff(signs) != 0)[0]
            for idx in sign_changes:
                x0, x1 = xs[idx], xs[idx + 1]
                y0, y1 = diff[idx], diff[idx + 1]
                if y1 == y0:
                    continue
                xi = float(x0 - y0 * (x1 - x0) / (y1 - y0))
                yi_arr = self._fit_fn(np.asarray([xi]))
                yi = float(yi_arr[0] if np.ndim(yi_arr) else yi_arr)
                key = (round(xi, 9), round(yi, 9))
                if key in seen:
                    continue
                seen.add(key)
                hits_x.append(xi)
                hits_y.append(yi)
                self.intersections_box.insert(
                    tk.END, f"{curve_label}: x={xi:.6g}, y={yi:.6g}"
                )
                found = True

        self.plot_controller.clear_crosses()
        if hits_x:
            self.plot_controller.draw_points(hits_x, hits_y, label="Intersections", style_kwargs={"color": "#1E88E5"})
        if not found:
            self.intersections_box.insert(tk.END, "No intersections in range.")

    def _overlay_literature(self) -> None:
        params: dict[str, float] = {}
        for name, var in self.lit_param_vars.items():
            text = var.get().strip()
            if not text:
                messagebox.showwarning("Literature", "Please provide parameter values.")
                return
            try:
                params[name] = float(text)
            except ValueError:
                messagebox.showwarning("Literature", f"Parameter '{name}' must be numeric.")
                return

        model = self.lit_model.get()
        if model == "Power" and params.get("a", 0.0) <= 0:
            messagebox.showwarning("Literature", "Parameter 'a' must be positive for the power model.")
            return

        fn, label = self._make_literature_function(model, params)

        xmin, xmax = self.axes.get_xlim()
        if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin == xmax:
            if self._current_xy is not None and not self._current_xy.empty:
                xs = self._current_xy["x"].to_numpy(dtype=float)
                xs = xs[np.isfinite(xs)]
                if xs.size:
                    xmin = float(np.nanmin(xs))
                    xmax = float(np.nanmax(xs))
        if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmin == xmax:
            messagebox.showwarning("Literature", "Plot data first to determine an X range.")
            return

        sample_x = np.linspace(xmin, xmax, 256)
        with np.errstate(all="ignore"):
            sample_y = np.asarray(fn(sample_x), dtype=float)
        mask = np.isfinite(sample_y)
        if not np.any(mask):
            messagebox.showwarning("Literature", "Curve is undefined for the current range.")
            return

        self._literature.append({"model": model, "params": params, "fn": fn, "label": label})
        self.plot_controller.draw_literature(
            sample_x[mask], sample_y[mask], label=label, fn=fn
        )
        self.plot_controller.clear_crosses()
        self.intersections_box.delete(0, tk.END)

    def _clear_literature(self) -> None:
        if self._literature:
            self._literature.clear()
        self.plot_controller.clear_literature()
        self.plot_controller.clear_crosses()
        self.intersections_box.delete(0, tk.END)

    def _refresh_inverse_params(self) -> None:
        previous = {name: var.get() for name, var in self.inv_param_vars.items()}
        for child in self.inv_params_frame.winfo_children():
            child.destroy()
        self.inv_param_vars.clear()

        model = self.inv_model.get()
        defaults = {
            "Linear": {"m": previous.get("m", "1.0"), "b": previous.get("b", "0.0")},
            "Quadratic": {
                "a": previous.get("a", "1.0"),
                "b": previous.get("b", "0.0"),
                "c": previous.get("c", "0.0"),
            },
            "Power": {"a": previous.get("a", "1.0"), "b": previous.get("b", "1.0")},
        }
        examples = {
            "Linear": "y = m·x + b",
            "Quadratic": "y = a·x² + b·x + c",
            "Power": "y = a·xᵇ (a>0)",
        }
        self.inv_example.configure(text=examples.get(model, ""))
        params = defaults.get(model, {})
        for row, (name, value) in enumerate(params.items()):
            ttk.Label(self.inv_params_frame, text=name).grid(row=row, column=0, sticky="w")
            var = tk.StringVar(value=value)
            ttk.Entry(self.inv_params_frame, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", padx=(6, 12), pady=2
            )
            self.inv_param_vars[name] = var

    def _on_inverse_solve(self) -> None:
        for item_id in self.inverse_table.get_children():
            self.inverse_table.delete(item_id)

        try:
            params = {name: float(var.get()) for name, var in self.inv_param_vars.items()}
        except (TypeError, ValueError):
            messagebox.showwarning("Inverse", "All parameters must be numeric.")
            return

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
            self.add_annotation_points(np.asarray(xs_to_plot, dtype=float), np.asarray(ys_to_plot, dtype=float), label="Inverse solutions")

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

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:.6g}"

    @staticmethod
    def _format_signed(value: float) -> str:
        sign = "+" if value >= 0 else "-"
        return f"{sign}{abs(value):.6g}"

    def _make_literature_function(
        self, model: str, params: dict[str, float]
    ) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
        if model == "Quadratic":
            a = float(params.get("a", 0.0))
            b = float(params.get("b", 0.0))
            c = float(params.get("c", 0.0))

            def fn(x: np.ndarray, A=a, B=b, C=c) -> np.ndarray:
                return A * x * x + B * x + C

            label = (
                f"Lit: y={self._format_number(a)}x^2 "
                f"{self._format_signed(b)}x {self._format_signed(c)}"
            )
            return fn, label

        if model == "Power":
            a = float(params.get("a", 0.0))
            b = float(params.get("b", 1.0))

            def fn(x: np.ndarray, A=a, B=b) -> np.ndarray:
                return A * np.power(x, B)

            label = f"Lit: y={self._format_number(a)}x^{self._format_number(b)}"
            return fn, label

        m = float(params.get("m", 1.0))
        b = float(params.get("b", 0.0))

        def fn(x: np.ndarray, M=m, B=b) -> np.ndarray:
            return M * x + B

        label = f"Lit: y={self._format_number(m)}x{self._format_signed(b)}"
        return fn, label

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
        if self.intersections_box.size() == 0:
            messagebox.showinfo("Copy", "Nothing to copy.")
            return
        rows = [self.intersections_box.get(idx) for idx in range(self.intersections_box.size())]
        self._copy_text("\n".join(rows))

    def _copy_residuals(self) -> None:
        if self._fit_fn is None or self._current_xy is None or self._current_xy.empty:
            messagebox.showinfo("Copy", "Residuals are not available.")
            return
        x_vals = self._current_xy["x"].to_numpy(dtype=float)
        try:
            predicted = np.asarray(self._fit_fn(x_vals), dtype=float)
        except Exception:
            messagebox.showinfo("Copy", "Unable to evaluate fitted model.")
            return
        export = self._current_xy.copy()
        export = export.assign(y_fit=predicted, residual=export["y"].to_numpy(dtype=float) - predicted)
        self._copy_text(export.to_csv(index=False))


__all__ = ["PlotPanel"]
