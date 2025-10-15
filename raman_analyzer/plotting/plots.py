"""Matplotlib plotting helpers for the Raman Analyzer application."""
from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvasQTAgg):
    """A convenience canvas embedding a Matplotlib figure."""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(6, 4))
        super().__init__(self.figure)
        self.axes = self.figure.add_subplot(111)

    def clear(self) -> None:
        """Clear the canvas and create a fresh axes instance."""

        self.figure.clf()
        self.axes = self.figure.add_subplot(111)


def _color_cycle(ax) -> Iterator[dict]:
    return iter(ax._get_lines.prop_cycler)


def draw_scatter(
    ax,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue: Optional[str] = "tag",
    jitter: bool = False,
) -> None:
    """Render a scatter plot grouped by hue."""

    ax.clear()
    if hue and hue in df.columns:
        cycler = _color_cycle(ax)
        for (tag, group) in df.groupby(hue):
            color = next(cycler, {}).get("color", None)
            x = group[x_col].to_numpy(dtype=float, copy=False)
            y = group[y_col].to_numpy(dtype=float, copy=False)
            if jitter and len(x) > 1:
                jitter_amount = 0.05 * np.std(x) if np.std(x) else 0.0
                x = x + np.random.normal(0, jitter_amount, size=len(x))
            ax.scatter(x, y, label=str(tag), color=color)
        ax.legend(title=hue)
    else:
        ax.scatter(df[x_col], df[y_col])
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def draw_line(ax, df: pd.DataFrame, x_col: str, y_col: str, hue: Optional[str] = "tag") -> None:
    """Render a line plot sorted by the x-column."""

    ax.clear()
    if hue and hue in df.columns:
        for (tag, group) in df.groupby(hue):
            sorted_group = group.sort_values(x_col)
            ax.plot(sorted_group[x_col], sorted_group[y_col], marker="o", label=str(tag))
        ax.legend(title=hue)
    else:
        sorted_df = df.sort_values(x_col)
        ax.plot(sorted_df[x_col], sorted_df[y_col], marker="o")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def draw_box(ax, df: pd.DataFrame, x_col: str = "tag", y_col: str = "value") -> None:
    """Render a box plot for the provided data."""

    ax.clear()
    groups = list(df.groupby(x_col))
    if not groups:
        return
    data = [group[y_col].dropna() for _, group in groups]
    labels = [str(name) for name, _ in groups]
    ax.boxplot(data, labels=labels)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def draw_violin(ax, df: pd.DataFrame, x_col: str = "tag", y_col: str = "value") -> None:
    """Render a violin plot for the provided data."""

    ax.clear()
    groups = list(df.groupby(x_col))
    if not groups:
        return
    data = [group[y_col].dropna() for _, group in groups]
    labels = [str(name) for name, _ in groups]
    ax.violinplot(data, showmeans=True, showextrema=True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)


def overlay_data_fit(ax, model_dict: dict, x_range: Tuple[float, float]) -> None:
    """Overlay the fitted data model onto the current axes."""

    if not model_dict:
        return
    xs = np.linspace(x_range[0], x_range[1], 200)
    model = model_dict.get("model")
    coeffs = model_dict.get("coeffs", ())
    label = None
    if model == "linear" and len(coeffs) == 2:
        ys = coeffs[0] * xs + coeffs[1]
        label = (
            f"Data fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f} "
            f"(R^2={model_dict.get('r2', float('nan')):.3f})"
        )
    elif model == "quadratic" and len(coeffs) == 3:
        ys = coeffs[0] * xs**2 + coeffs[1] * xs + coeffs[2]
        label = "Data quadratic fit"
    elif model == "power" and len(coeffs) == 2:
        ys = coeffs[0] * xs**coeffs[1]
        label = "Data power fit"
    else:
        return
    ax.plot(xs, ys, color="black", linestyle="--", label=label)
    ax.legend()


def overlay_literature(ax, model_dict: dict, x_range: Tuple[float, float]) -> None:
    """Overlay the literature model onto the current axes."""

    if not model_dict:
        return
    xs = np.linspace(x_range[0], x_range[1], 200)
    model = model_dict.get("model")
    coeffs = model_dict.get("coeffs", ())
    if model == "linear" and len(coeffs) == 2:
        ys = coeffs[0] * xs + coeffs[1]
    elif model == "quadratic" and len(coeffs) == 3:
        ys = coeffs[0] * xs**2 + coeffs[1] * xs + coeffs[2]
    elif model == "power" and len(coeffs) == 2:
        ys = coeffs[0] * xs**coeffs[1]
    else:
        return
    ax.plot(xs, ys, color="red", linestyle=":", label="Literature")
    ax.legend()


def mark_intersections(ax, points: List[Tuple[float, float]]) -> None:
    """Mark intersection points on the axes."""

    if not points:
        return
    xs, ys = zip(*points)
    ax.scatter(xs, ys, color="purple", marker="x", s=80, label="Intersections")
    ax.legend()
