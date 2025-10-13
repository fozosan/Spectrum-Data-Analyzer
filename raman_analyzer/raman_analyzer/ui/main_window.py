"""Main window implementation for the Raman Analyzer application."""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from raman_analyzer.analysis.trendlines import (
    fit_linear,
    intersections_linear_linear,
)
from raman_analyzer.io.loader import load_csvs
from raman_analyzer.models.session import AnalysisSession
from raman_analyzer.plotting.plots import (
    PlotCanvas,
    draw_box,
    draw_scatter,
    overlay_data_fit,
    overlay_literature,
    mark_intersections,
)
from raman_analyzer.ui.widgets.calc_builder import CalcBuilderWidget
from raman_analyzer.ui.widgets.data_table import DataTableWidget
from raman_analyzer.ui.widgets.file_list import FileListWidget
from raman_analyzer.ui.widgets.plot_config import PlotConfigWidget
from raman_analyzer.ui.widgets.trendline_form import TrendlineForm


class MainWindow(QMainWindow):
    """Primary window managing UI interactions and session state."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Raman Analyzer")
        self.resize(1200, 800)

        self.session = AnalysisSession()
        self.current_plot_config: Optional[dict] = None
        self.current_plot_data: Optional[pd.DataFrame] = None

        self._create_actions()
        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------ UI setup
    def _create_actions(self) -> None:
        open_action = QAction("Load CSVs", self)
        open_action.triggered.connect(self._load_csvs)
        self.toolbar = self.addToolBar("Main")
        self.toolbar.addAction(open_action)

    def _setup_ui(self) -> None:
        central = QWidget(self)
        central_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal, central)

        # Left pane: file list + data table
        left_widget = QWidget(splitter)
        left_layout = QVBoxLayout(left_widget)
        self.file_list = FileListWidget(left_widget)
        self.data_table = DataTableWidget(left_widget)
        left_layout.addWidget(self.file_list)
        left_layout.addWidget(self.data_table)

        # Right pane: controls + plot
        right_widget = QWidget(splitter)
        right_layout = QVBoxLayout(right_widget)
        self.calc_builder = CalcBuilderWidget(right_widget)
        self.plot_config = PlotConfigWidget(right_widget)
        self.trendline_form = TrendlineForm(right_widget)
        self.canvas = PlotCanvas()
        self.toolbar_canvas = NavigationToolbar2QT(self.canvas, right_widget)

        right_layout.addWidget(self.calc_builder)
        right_layout.addWidget(self.plot_config)
        right_layout.addWidget(self.trendline_form)
        right_layout.addWidget(self.toolbar_canvas)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        central_layout.addWidget(splitter)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))

    def _connect_signals(self) -> None:
        self.file_list.tagChanged.connect(self._on_tag_changed)
        self.file_list.selectionChanged.connect(self._on_file_selected)
        self.calc_builder.metricComputed.connect(self._on_metric_computed)
        self.plot_config.plotRequested.connect(self._on_plot_requested)
        self.plot_config.exportMetricsRequested.connect(self._export_metrics)
        self.plot_config.exportPlotRequested.connect(self._export_plot)
        self.trendline_form.fitRequested.connect(self._on_fit_requested)
        self.trendline_form.literatureOverlayRequested.connect(self._on_literature_overlay)
        self.trendline_form.intersectionsRequested.connect(self._on_intersections_requested)
        self.trendline_form.exportIntersectionsRequested.connect(
            self._export_intersections
        )

    # ------------------------------------------------------------------ Loading data
    def _load_csvs(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select peak CSV files",
            "",
            "CSV Files (*.csv)",
        )
        if not paths:
            return
        df = load_csvs(paths)
        if df.empty:
            QMessageBox.warning(self, "Load CSVs", "No data found in selected files.")
            return
        self.session.set_raw_data(df)
        files = df["file"].dropna().unique().tolist()
        self.file_list.set_files(files, self.session.file_to_tag)
        self.calc_builder.set_data(df, self.session.file_to_tag)
        available_attrs = [
            col
            for col in ["area", "height", "fwhm", "center", "area_pct"]
            if col in df.columns
        ]
        if not available_attrs:
            available_attrs = ["area", "height", "fwhm", "center", "area_pct"]
        self.calc_builder.set_available_attributes(available_attrs)
        self._update_plot_metrics()
        self._on_file_selected(files[:1])
        self.statusBar().showMessage(f"Loaded {len(files)} files", 5000)

    # ------------------------------------------------------------------ File interactions
    def _on_file_selected(self, files: List[str]) -> None:
        if not files:
            self.data_table.set_dataframe(pd.DataFrame())
            return
        if self.session.raw_df.empty:
            return
        subset = self.session.raw_df[self.session.raw_df["file"].isin(files)]
        self.data_table.set_dataframe(subset)

    def _on_tag_changed(self, file_id: str, tag: str) -> None:
        self.session.set_tag(file_id, tag)
        self.calc_builder.set_data(self.session.raw_df, self.session.file_to_tag)
        self._update_plot_metrics()

    # ------------------------------------------------------------------ Metrics handling
    def _on_metric_computed(self, metric_name: str, long_df: pd.DataFrame) -> None:
        per_file = long_df[["file", "value"]]
        self.session.update_metric(metric_name, per_file)
        self._update_plot_metrics()
        self.statusBar().showMessage(f"Metric '{metric_name}' computed", 5000)

    def _update_plot_metrics(self) -> None:
        df = self.session.results_df
        metrics = [col for col in df.columns if col not in {"file", "tag"}]
        self.plot_config.set_metrics(metrics)

    def _long_results(self) -> pd.DataFrame:
        df = self.session.results_df
        if df.empty:
            return pd.DataFrame(columns=["file", "tag", "metric_name", "value"])
        metrics = [col for col in df.columns if col not in {"file", "tag"}]
        if not metrics:
            return pd.DataFrame(columns=["file", "tag", "metric_name", "value"])
        long_df = df.melt(
            id_vars=["file", "tag"],
            value_vars=metrics,
            var_name="metric_name",
            value_name="value",
        )
        return long_df

    # ------------------------------------------------------------------ Plotting
    def _on_plot_requested(self, config: dict) -> None:
        df = self.session.results_df
        if df.empty:
            return
        y_metric = config.get("y_axis")
        if y_metric not in df.columns:
            return
        x_axis = config.get("x_axis")
        plot_type = config.get("plot_type")

        if plot_type == "Box":
            long_df = self._long_results()
            metric_df = long_df[long_df["metric_name"] == y_metric]
            draw_box(self.canvas.axes, metric_df, x_col="tag", y_col="value")
            self.canvas.axes.set_ylabel(y_metric)
            self.current_plot_data = None
        else:
            if x_axis == "Group (categorical)":
                plot_df = df[["tag", y_metric]].dropna()
                if plot_df.empty:
                    return
                categories = pd.Categorical(plot_df["tag"].fillna("(untagged)"))
                numeric_x = categories.codes.astype(float)
                plot_df = plot_df.assign(tag=categories, x=numeric_x, y=plot_df[y_metric])
                draw_scatter(self.canvas.axes, plot_df, "x", "y", hue="tag")
                self.canvas.axes.set_xticks(range(len(categories.categories)))
                self.canvas.axes.set_xticklabels(categories.categories)
                self.canvas.axes.set_xlabel("Group")
                self.canvas.axes.set_ylabel(y_metric)
            else:
                if x_axis not in df.columns:
                    return
                plot_df = df[["tag", x_axis, y_metric]].dropna()
                if plot_df.empty:
                    return
                plot_df = plot_df.rename(columns={x_axis: "x", y_metric: "y"})
                draw_scatter(self.canvas.axes, plot_df, "x", "y", hue="tag")
                self.canvas.axes.set_xlabel(x_axis)
                self.canvas.axes.set_ylabel(y_metric)
            self.current_plot_data = plot_df[["tag", "x", "y"]]
            if config.get("error_bars") and not self.current_plot_data.empty:
                self._add_error_bars(x_axis, y_metric)

        self.current_plot_config = config
        self.canvas.draw()
        self._overlay_trendlines()

        x_limits = config.get("x_limits", (None, None))
        y_limits = config.get("y_limits", (None, None))
        if x_limits[0] is not None or x_limits[1] is not None:
            self.canvas.axes.set_xlim(
                x_limits[0] if x_limits[0] is not None else self.canvas.axes.get_xlim()[0],
                x_limits[1] if x_limits[1] is not None else self.canvas.axes.get_xlim()[1],
            )
        if y_limits[0] is not None or y_limits[1] is not None:
            self.canvas.axes.set_ylim(
                y_limits[0] if y_limits[0] is not None else self.canvas.axes.get_ylim()[0],
                y_limits[1] if y_limits[1] is not None else self.canvas.axes.get_ylim()[1],
            )
        self.canvas.draw()

    def _add_error_bars(self, x_axis: str, y_metric: str) -> None:
        if self.current_plot_data is None or self.current_plot_data.empty:
            return
        df = self.current_plot_data
        if "tag" not in df.columns or "x" not in df.columns:
            return
        grouped = df.groupby("tag")
        for tag, group in grouped:
            x_mean = group["x"].mean()
            y_mean = group["y"].mean()
            y_std = group["y"].std(ddof=0) if len(group) > 1 else 0
            self.canvas.axes.errorbar(x_mean, y_mean, yerr=y_std, fmt="o", color="black")

    def _overlay_trendlines(self) -> None:
        if self.current_plot_data is None or self.current_plot_data.empty:
            return
        x_min = float(self.current_plot_data["x"].min())
        x_max = float(self.current_plot_data["x"].max())
        overlay_data_fit(self.canvas.axes, self.session.data_fit, (x_min, x_max))
        overlay_literature(self.canvas.axes, self.session.literature_fit, (x_min, x_max))
        mark_intersections(self.canvas.axes, self.session.intersections)
        self.canvas.draw()

    # ------------------------------------------------------------------ Trendlines
    def _on_fit_requested(self, model: str) -> None:
        if self.current_plot_data is None or self.current_plot_data.empty:
            QMessageBox.information(
                self, "Trendline", "Plot data is required before fitting."
            )
            return
        if model != "linear":
            QMessageBox.information(
                self, "Trendline", "Only linear fits are supported in this version."
            )
            return
        x = self.current_plot_data["x"].to_numpy(dtype=float)
        y = self.current_plot_data["y"].to_numpy(dtype=float)
        if len(x) < 2:
            QMessageBox.warning(self, "Trendline", "Not enough data points to fit.")
            return
        fit = fit_linear(x, y)
        self.session.set_data_fit(fit)
        summary = f"Linear fit: y={fit['coeffs'][0]:.3f}x+{fit['coeffs'][1]:.3f} (R^2={fit['r2']:.3f})"
        self.trendline_form.set_fit_summary(summary)
        if self.current_plot_config:
            self._on_plot_requested(self.current_plot_config)

    def _on_literature_overlay(self, payload: dict) -> None:
        if payload.get("model") != "linear" or len(payload.get("coeffs", ())) != 2:
            QMessageBox.information(
                self, "Trendline", "Only linear literature models are supported."
            )
            return
        self.session.set_literature_fit(payload)
        if self.current_plot_config:
            self._on_plot_requested(self.current_plot_config)

    def _on_intersections_requested(self) -> None:
        if not self.session.data_fit or not self.session.literature_fit:
            QMessageBox.information(
                self,
                "Intersections",
                "Both data and literature fits are required to compute intersections.",
            )
            return
        data_coeffs = self.session.data_fit.get("coeffs", ())
        lit_coeffs = self.session.literature_fit.get("coeffs", ())
        if (
            self.session.data_fit.get("model") == "linear"
            and self.session.literature_fit.get("model") == "linear"
        ):
            points = intersections_linear_linear(
                data_coeffs[0], data_coeffs[1], lit_coeffs[0], lit_coeffs[1]
            )
        else:
            points = []
        self.session.set_intersections(points)
        self.trendline_form.set_intersections(points)
        if self.current_plot_config:
            self._on_plot_requested(self.current_plot_config)

    # ------------------------------------------------------------------ Export helpers
    def _export_metrics(self) -> None:
        long_df = self._long_results()
        if long_df.empty:
            QMessageBox.information(self, "Export Metrics", "No metrics to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export metrics CSV",
            "metrics.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        long_df.to_csv(path, index=False)
        self.statusBar().showMessage(f"Metrics exported to {path}", 5000)

    def _export_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            "plot.png",
            "PNG Files (*.png)",
        )
        if not path:
            return
        self.canvas.figure.savefig(path, dpi=300)
        self.statusBar().showMessage(f"Plot saved to {path}", 5000)

    def _export_intersections(self) -> None:
        points = self.session.intersections
        if not points:
            QMessageBox.information(
                self, "Export Intersections", "No intersections to export."
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export intersections CSV",
            "intersections.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        df = pd.DataFrame(points, columns=["x", "y"])
        df.to_csv(path, index=False)
        self.statusBar().showMessage(f"Intersections exported to {path}", 5000)
