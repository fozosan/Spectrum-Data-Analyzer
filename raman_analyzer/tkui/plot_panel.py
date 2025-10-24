"""Plot controls and chart rendering for the Tkinter UI."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Sequence

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
    intersections_linear_linear,
    intersections_numeric,
    intersections_poly_linear,
)


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

        export_row = ttk.Frame(control_box)
        export_row.grid(row=1, column=0, columnspan=10, sticky="w", pady=(4, 0))
        ttk.Button(export_row, text="Export XY", command=self._export_xy).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Plot (PNG)", command=self._export_plot).pack(side="left", padx=2)
        ttk.Button(export_row, text="Export Group Stats", command=self._export_group_stats).pack(side="left", padx=2)

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

        self.fit_summary = tk.Text(fit_box, height=4, width=60)
        self.fit_summary.grid(row=1, column=0, columnspan=3, sticky="ew", padx=2, pady=2)
        self.fit_summary.configure(state="disabled")

        self.intersections_box = tk.Listbox(fit_box, height=5)
        self.intersections_box.grid(row=2, column=0, columnspan=3, sticky="ew", padx=2, pady=(2, 6))
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
        self.lit_a = tk.StringVar(value="1.0")
        self.lit_b = tk.StringVar(value="0.0")
        self.lit_c = tk.StringVar(value="0.0")
        ttk.Label(literature_box, text="a/m").grid(row=1, column=0, sticky="w")
        ttk.Entry(literature_box, textvariable=self.lit_a, width=10).grid(row=1, column=1, sticky="w", padx=4)
        ttk.Label(literature_box, text="b").grid(row=1, column=2, sticky="w")
        ttk.Entry(literature_box, textvariable=self.lit_b, width=10).grid(row=1, column=3, sticky="w", padx=4)
        ttk.Label(literature_box, text="c").grid(row=1, column=4, sticky="w")
        ttk.Entry(literature_box, textvariable=self.lit_c, width=10).grid(row=1, column=5, sticky="w", padx=4)
        literature_buttons = ttk.Frame(literature_box)
        literature_buttons.grid(row=2, column=0, columnspan=6, sticky="w", pady=(4, 2))
        ttk.Button(literature_buttons, text="Overlay", command=self._overlay_literature).pack(side="left", padx=2)
        ttk.Button(literature_buttons, text="Clear", command=self._clear_literature).pack(side="left", padx=2)
        ttk.Button(
            literature_buttons,
            text="Find Intersections",
            command=self._on_intersections,
        ).pack(side="left", padx=2)

        # --------------------------- figure ---------------------------
        self.figure = Figure(figsize=(6, 4), dpi=120)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True, padx=6, pady=(0, 6))

        # --------------------------- state ---------------------------
        self._current_xy = pd.DataFrame(columns=["file", "tag", "x", "y", "__group__"])
        self._fit: dict[str, object] | None = None
        self._literature: list[dict[str, object]] = []

    # ------------------------------------------------------------------ public API
    def set_metrics_for_xy(self, names: Sequence[str]) -> None:
        metrics = [str(name) for name in names if name]

        choices: list[str] = ["X Mapping", "Tag (numeric)"]
        choices.extend(metrics)

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
        ]
        _ensure_selection(self.x_field, self.x_combo, x_fallbacks)
        x_choice = self.x_field.get()

        preferred_y = [name for name in ("Selection A", "Selection B") if name in values]
        y_fallbacks = [
            self.y_field.get(),
            *preferred_y,
            *[item for item in values if item not in {"X Mapping", "Tag (numeric)", x_choice}],
            *[item for item in ("X Mapping", "Tag (numeric)") if item != x_choice],
        ]
        _ensure_selection(self.y_field, self.y_combo, y_fallbacks)

    def set_metrics(self, names: Sequence[str]) -> None:
        self.set_metrics_for_xy(names)

    # ------------------------------------------------------------------ plotting helpers
    def _build_xy(self, x_label: str, y_label: str) -> pd.DataFrame | None:
        df = self.session.results_df
        if df is None or df.empty:
            return None

        work = df.copy()
        work["x"] = self._resolve_axis(x_label, work)
        work["y"] = self._resolve_axis(y_label, work)
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

        series = work.get(label)
        if series is None:
            return pd.Series(np.nan, index=work.index)
        return pd.to_numeric(series, errors="coerce")

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
        self.axes.clear()

        for label, group_df in work.groupby("__group__"):
            xs = group_df["x"].to_numpy(dtype=float)
            ys = group_df["y"].to_numpy(dtype=float)
            if self.plot_type.get() == "Line":
                order = np.argsort(xs)
                xs = xs[order]
                ys = ys[order]
                self.axes.plot(xs, ys, label=label if label and label != "All" else None)
            else:
                self.axes.scatter(xs, ys, label=label if label and label != "All" else None)

        if self.show_err.get():
            renamed = work.rename(columns={"__group__": "tag"})
            grouped = compute_error_table(renamed, mode="SEM")
            if not grouped.empty:
                for grp_label, grp_df in grouped.groupby("tag"):
                    xs = grp_df["x"].to_numpy(dtype=float)
                    means = grp_df["mean"].to_numpy(dtype=float)
                    yerr = grp_df.get("yerr")
                    err = None if yerr is None else yerr.to_numpy(dtype=float)
                    self.axes.errorbar(
                        xs,
                        means,
                        yerr=err,
                        fmt="o",
                        capsize=2,
                        alpha=0.7,
                        label=None if grp_label in ("", "All") else f"{grp_label} (±SEM)",
                    )

        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.grid(True, linestyle="--", alpha=0.3)

        if self._fit is not None and not work.empty:
            try:
                xmin = float(np.nanmin(work["x"]))
                xmax = float(np.nanmax(work["x"]))
                if np.isfinite(xmin) and np.isfinite(xmax) and xmin != xmax:
                    x_grid = np.linspace(xmin, xmax, 256)
                    y_grid = self._eval_model(self._fit["model"], self._fit["coeffs"], x_grid)
                    self.axes.plot(
                        x_grid,
                        y_grid,
                        color="black",
                        linewidth=2,
                        alpha=0.7,
                        label=f"Fit: {self._fit['model']}",
                    )
            except Exception:
                pass

        for entry in self._literature:
            entry.pop("artist", None)
            try:
                xmin, xmax = self.axes.get_xlim()
                if not (np.isfinite(xmin) and np.isfinite(xmax)):
                    continue
                x_grid = np.linspace(xmin, xmax, 256)
                y_grid = self._eval_model(entry["model"], entry["coeffs"], x_grid)
                (line,) = self.axes.plot(
                    x_grid,
                    y_grid,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"Lit: {entry['model']}",
                )
                entry["artist"] = line
            except Exception:
                continue

        _, labels = self.axes.get_legend_handles_labels()
        if labels:
            self.axes.legend(loc="best")

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
        try:
            predicted = self._eval_model(
                self._fit["model"],
                self._fit["coeffs"],
                self._current_xy["x"].to_numpy(dtype=float),
            )
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
            elif self.fit_model.get() == "Quadratic":
                result = fit_quadratic(x, y)
                coeffs = result["coeffs"]
                model_name = "Quadratic"
            else:
                result = fit_power(x, y)
                coeffs = result["coeffs"]
                model_name = "Power"
        except Exception as exc:
            messagebox.showwarning("Fit", f"Unable to compute fit: {exc}")
            return

        self._fit = {
            "model": model_name,
            "coeffs": tuple(float(c) for c in coeffs),
            "r2": float(result.get("r2", np.nan)),
        }

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

    def _on_intersections(self) -> None:
        self.intersections_box.delete(0, tk.END)
        if self._fit is None:
            messagebox.showinfo(
                "Intersections",
                "No trendline has been fitted yet.\n\nFit a trendline to the current data before finding intersections.",
            )
            self.intersections_box.insert(tk.END, "Fit required first.")
            return
        if not self._literature:
            self.intersections_box.insert(tk.END, "No literature overlays.")
            return
        if self._current_xy is None or self._current_xy.empty:
            self.intersections_box.insert(tk.END, "Nothing plotted.")
            return

        xmin = float(np.nanmin(self._current_xy["x"]))
        xmax = float(np.nanmax(self._current_xy["x"]))
        if not (np.isfinite(xmin) and np.isfinite(xmax)):
            self.intersections_box.insert(tk.END, "Invalid plot range.")
            return

        found = False
        for entry in self._literature:
            try:
                for x_val, y_val in self._pairwise_intersections(self._fit, entry, xmin, xmax):
                    self.intersections_box.insert(tk.END, f"x={x_val:.6g}, y={y_val:.6g}")
                    found = True
            except Exception:
                continue

        if not found:
            self.intersections_box.insert(tk.END, "No intersections in range.")

    def _overlay_literature(self) -> None:
        try:
            a = float(self.lit_a.get() or 0)
            b = float(self.lit_b.get() or 0)
            c = float(self.lit_c.get() or 0)
        except ValueError:
            messagebox.showwarning("Literature", "Invalid coefficients.")
            return

        model = self.lit_model.get()
        if model == "Quadratic":
            coeffs: tuple[float, ...] = (a, b, c)
        else:
            coeffs = (a, b)

        entry: dict[str, object] = {"model": model, "coeffs": coeffs}
        self._literature.append(entry)

        if self._current_xy is not None and not self._current_xy.empty:
            try:
                xmin, xmax = self.axes.get_xlim()
                if not (np.isfinite(xmin) and np.isfinite(xmax)):
                    xmin = float(np.nanmin(self._current_xy["x"]))
                    xmax = float(np.nanmax(self._current_xy["x"]))
                x_grid = np.linspace(xmin, xmax, 256)
                y_grid = self._eval_model(model, coeffs, x_grid)
                (line,) = self.axes.plot(
                    x_grid,
                    y_grid,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"Lit: {model}",
                )
                entry["artist"] = line
                _, labels = self.axes.get_legend_handles_labels()
                if labels:
                    self.axes.legend(loc="best")
                self.canvas.draw_idle()
            except Exception:
                pass

    def _clear_literature(self) -> None:
        for entry in self._literature:
            artist = entry.get("artist")
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
        self._literature.clear()
        self.canvas.draw_idle()

    def _eval_model(self, model: str, coeffs: tuple[float, ...], x_values: np.ndarray) -> np.ndarray:
        if model == "Linear":
            m, b = coeffs
            return eval_linear(x_values, m, b)
        if model == "Quadratic":
            a, b, c = coeffs
            return eval_quadratic(x_values, a, b, c)
        a, b = coeffs
        return eval_power(x_values, a, b)

    def _pairwise_intersections(
        self,
        fit_info: dict[str, object],
        literature: dict[str, object],
        xmin: float,
        xmax: float,
    ) -> list[tuple[float, float]]:
        model_fit = fit_info["model"]
        coeffs_fit = fit_info["coeffs"]
        model_lit = literature["model"]
        coeffs_lit = literature["coeffs"]

        if model_fit == "Linear" and model_lit == "Linear":
            slope_a, intercept_a = coeffs_fit
            slope_b, intercept_b = coeffs_lit
            points = intersections_linear_linear(slope_a, intercept_a, slope_b, intercept_b)
        elif (model_fit == "Quadratic" and model_lit == "Linear") or (
            model_fit == "Linear" and model_lit == "Quadratic"
        ):
            if model_fit == "Quadratic":
                a, b, c = coeffs_fit
                slope, intercept = coeffs_lit
            else:
                a, b, c = coeffs_lit
                slope, intercept = coeffs_fit
            points = intersections_poly_linear(a, b, c, slope, intercept)
        else:
            def fit_eval(x_val: float) -> float:
                return float(self._eval_model(model_fit, coeffs_fit, np.array([x_val]))[0])

            def lit_eval(x_val: float) -> float:
                return float(self._eval_model(model_lit, coeffs_lit, np.array([x_val]))[0])

            numeric_points = intersections_numeric(fit_eval, lit_eval, xmin, xmax)
            points = [(x_val, fit_eval(x_val)) for x_val, _ in numeric_points]

        filtered: list[tuple[float, float]] = []
        for x_val, y_val in points:
            if np.isfinite(x_val) and np.isfinite(y_val) and xmin <= x_val <= xmax:
                filtered.append((float(x_val), float(y_val)))
        return filtered


__all__ = ["PlotPanel"]
