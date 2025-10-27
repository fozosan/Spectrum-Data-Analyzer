"""Plot controls and chart rendering for the Tkinter UI."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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

        # --------------------------- controls ---------------------------
        self.controls_container = ttk.Frame(controls_parent)
        self.controls_container.pack(side="top", fill="x", expand=True)

        control_box = ttk.LabelFrame(self.controls_container, text="Plot")
        control_box.pack(side="top", fill="x", padx=6, pady=6)

        self.x_field = tk.StringVar(value="X Mapping")
        self.y_field = tk.StringVar(value="")
        self.group_field = tk.StringVar(value="Tag")
        self.plot_type = tk.StringVar(value="Scatter")
        self.show_err = tk.BooleanVar(value=False)
        self.x_label_text = tk.StringVar(value="")
        self.y_label_text = tk.StringVar(value="")
        self.auto_x = tk.BooleanVar(value=True)
        self.auto_y = tk.BooleanVar(value=True)
        # Where to adjust defaults later: tweak auto range defaults, entry widths, or combo selections.
        self._derived_metrics: tuple[str, ...] = ()

        ttk.Label(control_box, text="X").grid(row=0, column=0, sticky="w")
        self.x_combo = ttk.Combobox(control_box, textvariable=self.x_field, state="readonly", width=18)
        self.x_combo["values"] = ("X Mapping", "Tag (numeric)")
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

        ttk.Checkbutton(control_box, text="Error bars", variable=self.show_err).grid(
            row=0, column=8, padx=4, pady=2, sticky="w"
        )

        ttk.Button(control_box, text="Plot", command=self._on_plot).grid(row=0, column=9, padx=6, pady=2, sticky="e")

        label_row = ttk.Frame(control_box)
        label_row.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(4, 0))
        ttk.Label(label_row, text="X label").pack(side="left")
        self.x_label_entry = ttk.Entry(label_row, textvariable=self.x_label_text, width=20)
        self.x_label_entry.pack(side="left", padx=(4, 12))
        ttk.Label(label_row, text="Y label").pack(side="left")
        self.y_label_entry = ttk.Entry(label_row, textvariable=self.y_label_text, width=20)
        self.y_label_entry.pack(side="left", padx=(4, 0))

        range_row = ttk.Frame(control_box)
        range_row.grid(row=2, column=0, columnspan=10, sticky="ew", pady=(4, 0))
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
        export_row.grid(row=3, column=0, columnspan=10, sticky="w", pady=(4, 0))
        ttk.Button(export_row, text="Export XY", command=self._export_xy).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Plot (PNG)", command=self._export_plot).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Group Stats", command=self._export_group_stats).pack(side="left", padx=2)

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
        self.fit_summary.grid(row=1, column=0, columnspan=4, sticky="ew", padx=2, pady=2)
        self.fit_summary.configure(state="disabled")

        self.intersections_box = tk.Listbox(fit_box, height=5)
        self.intersections_box.grid(row=2, column=0, columnspan=4, sticky="ew", padx=2, pady=(2, 6))
        ttk.Button(fit_box, text="Export Intersections", command=self._export_intersections).grid(
            row=3, column=0, padx=2, pady=2, sticky="w"
        )
        ttk.Button(fit_box, text="Export Residuals", command=self._export_residuals).grid(
            row=3, column=1, padx=2, pady=2, sticky="w"
        )

        literature_box = ttk.LabelFrame(fit_box, text="Literature Overlay")
        literature_box.grid(row=4, column=0, columnspan=3, sticky="ew", padx=2, pady=(6, 2))
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

        choices: list[str] = ["X Mapping", "Tag (numeric)"]
        choices.extend(metrics)
        choices.extend(derived)

        values = tuple(choices)
        self.x_combo["values"] = values
        self.y_combo["values"] = values

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
            "X Mapping",
            "Tag (numeric)",
            *metrics,
            *derived,
        ]
        _ensure_selection(self.x_field, self.x_combo, x_fallbacks)
        x_choice = self.x_field.get()

        preferred_y = [name for name in ["Selection A", "Selection B", *derived] if name in values]
        y_fallbacks = [
            self.y_field.get(),
            *preferred_y,
            *[item for item in values if item not in {"X Mapping", "Tag (numeric)", x_choice}],
            *[item for item in ("X Mapping", "Tag (numeric)") if item != x_choice],
        ]
        _ensure_selection(self.y_field, self.y_combo, y_fallbacks)

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
        if label == "X Mapping":
            mapping = dict(getattr(self.session, "x_mapping", {}) or {})
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
            if label in {"X Mapping", "Tag (numeric)"}:
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

        x_axis_label = self.x_label_text.get().strip() or x_label
        y_axis_label = self.y_label_text.get().strip() or y_label

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

        error_entries: list[dict[str, object]] = []
        if self.show_err.get():
            renamed = work.rename(columns={"__group__": "tag"})
            grouped = compute_error_table(renamed, mode="SEM")
            if not grouped.empty:
                for grp_label, grp_df in grouped.groupby("tag"):
                    xs = grp_df["x"].to_numpy(dtype=float)
                    means = grp_df["mean"].to_numpy(dtype=float)
                    yerr = grp_df.get("yerr")
                    err = None if yerr is None else yerr.to_numpy(dtype=float)
                    legend = None if grp_label in ("", "All") else f"{grp_label} (±SEM)"
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
        renamed = self._current_xy.rename(columns={"__group__": "tag"})
        grouped = compute_error_table(renamed, mode="SEM")
        if grouped.empty:
            messagebox.showinfo("Export Group Stats", "No grouped statistics available.")
            return
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


__all__ = ["PlotPanel"]
