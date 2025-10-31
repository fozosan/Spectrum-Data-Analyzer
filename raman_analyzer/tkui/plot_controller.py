"""Helper utilities for drawing overlays on the Matplotlib canvas."""
from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PlotController:
    """Utility class that manages fit and overlay artists on a plot."""

    def __init__(self, canvas: FigureCanvasTkAgg, axes: Axes) -> None:
        self.canvas = canvas
        self.ax = axes
        self._fit_line = None
        self._literature_lines: list = []
        self._cross_markers: list = []

    # ------------------------------------------------------------------ base data
    def draw_scatter(
        self,
        series: Iterable[dict[str, object]],
        *,
        x_label: str = "",
        y_label: str = "",
        xlim: Optional[tuple[float, float]] = None,
        ylim: Optional[tuple[float, float]] = None,
        plot_type: str = "scatter",
        error_series: Optional[Iterable[dict[str, object]]] = None,
    ) -> None:
        """Render the base plot for the provided series on the axes."""

        self.ax.clear()

        plot_mode = (plot_type or "scatter").strip().lower()
        legend_labels: list[str] = []

        for entry in series:
            xs = np.asarray(entry.get("x", []), dtype=float)
            ys = np.asarray(entry.get("y", []), dtype=float)
            if xs.size == 0 or ys.size == 0:
                continue
            label = entry.get("label")
            if plot_mode == "line":
                order = np.argsort(xs)
                xs = xs[order]
                ys = ys[order]
                self.ax.plot(xs, ys, label=label)
            else:
                self.ax.scatter(xs, ys, label=label)
            if label:
                legend_labels.append(str(label))

        if error_series:
            for overlay in error_series:
                xs = np.asarray(overlay.get("x", []), dtype=float)
                means = np.asarray(overlay.get("mean", []), dtype=float)
                if xs.size == 0 or means.size == 0:
                    continue
                yerr = overlay.get("yerr")
                label = overlay.get("label")
                err_args = {}
                if yerr is not None:
                    err_args["yerr"] = np.asarray(yerr, dtype=float)
                self.ax.errorbar(
                    xs,
                    means,
                    fmt="o",
                    capsize=2,
                    alpha=0.7,
                    label=label,
                    **err_args,
                )
                if label:
                    legend_labels.append(str(label))

        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.grid(True, linestyle="--", alpha=0.3)

        if xlim is not None:
            self.ax.set_xlim(*xlim)
        if ylim is not None:
            self.ax.set_ylim(*ylim)

        if legend_labels:
            self.ax.legend(loc="best")

        self.canvas.draw_idle()

    # ------------------------------------------------------------------ fits
    def clear_fit(self) -> None:
        """Remove the currently rendered fit line, if present."""

        if self._fit_line is not None:
            try:
                self._fit_line.remove()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
            self._fit_line = None

    def draw_fit(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        x_data: Optional[Sequence[float]] = None,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        label: str = "Fit",
    ) -> None:
        """Draw a fitted line over the current axes using the provided function."""

        lo: Optional[float] = x_min
        hi: Optional[float] = x_max
        if lo is None or hi is None:
            data = np.asarray(list(x_data) if x_data is not None else [])
            if data.size:
                lo = float(np.nanmin(data))
                hi = float(np.nanmax(data))
            else:
                lo, hi = self.ax.get_xlim()
        if lo is None or hi is None:
            return
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            lo, hi = self.ax.get_xlim()
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            return

        xs = np.linspace(lo, hi, 256)
        ys = fn(xs)

        self.clear_fit()
        (self._fit_line,) = self.ax.plot(xs, ys, lw=2.0, alpha=0.95, label=label)
        self._update_legend()
        self.canvas.draw_idle()

    # ---------------------------------------------------------------- overlays
    def clear_literature(self) -> None:
        for artist in self._literature_lines:
            try:
                artist.remove()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        self._literature_lines.clear()
        self.canvas.draw_idle()

    def draw_literature(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        *,
        label: str = "Literature",
        style_kwargs: Optional[dict] = None,
        fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        kwargs = {"linestyle": "--", "linewidth": 1.8, "alpha": 0.85}
        if style_kwargs:
            kwargs.update(style_kwargs)
        (line,) = self.ax.plot(xs, ys, label=label, **kwargs)
        # Preserve context on the line artist so later consumers (intersections)
        # can resample the analytic curve rather than relying on plotted points.
        setattr(line, "_literature_fn", fn)
        setattr(line, "_literature_label", label)
        self._literature_lines.append(line)
        self._update_legend()
        self.canvas.draw_idle()

    def clear_crosses(self) -> None:
        for artist in self._cross_markers:
            try:
                artist.remove()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        self._cross_markers.clear()
        self.canvas.draw_idle()

    def draw_points(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        *,
        label: Optional[str] = None,
        style_kwargs: Optional[dict] = None,
    ):
        style = {"s": 36, "marker": "x", "linewidths": 1.0, "zorder": 4}
        if style_kwargs:
            style.update(style_kwargs)
        scatter = self.ax.scatter(xs, ys, label=label, **style)
        self._cross_markers.append(scatter)
        # Leave legend handling to the caller so overlays can opt-in explicitly.
        self.canvas.draw_idle()
        return scatter

    # ----------------------------------------------------------------- helpers
    def _update_legend(self) -> None:
        handles, labels = self.ax.get_legend_handles_labels()
        if labels:
            self.ax.legend(loc="best")

