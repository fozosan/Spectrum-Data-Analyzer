"""Main window implementation for the Raman Analyzer application."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from PyQt5.QtCore import Qt, QThread, pyqtSlot
from PyQt5.QtWidgets import (
    QAction,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

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
from raman_analyzer.io.session_io import load_session, save_session
from raman_analyzer.models.session import AnalysisSession
from raman_analyzer.plotting.plots import (
    PlotCanvas,
    draw_line,
    draw_scatter,
    overlay_data_fit,
    overlay_literature,
    mark_intersections,
)
from raman_analyzer.ui.widgets.calc_builder import CalcBuilderWidget
from raman_analyzer.ui.widgets.data_table import DataTableWidget
from raman_analyzer.ui.widgets.file_list import FileListWidget
from raman_analyzer.ui.widgets.plot_config import PlotConfigWidget
from raman_analyzer.ui.widgets.selection_panel import SelectionPanel
from raman_analyzer.ui.widgets.trendline_form import TrendlineForm
from raman_analyzer.ui.workers import CsvLoaderWorker


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
        self._loader_thread: Optional[QThread] = None
        self._loader_worker: Optional[CsvLoaderWorker] = None
        self._load_failed: bool = False
        self._current_grid_file: Optional[str] = None

    # ------------------------------------------------------------------ UI setup
    def _create_actions(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        open_action = QAction("Load CSVs", self)
        open_action.triggered.connect(self._load_csvs)
        import_map_action = QAction("Import Tags/X CSV", self)
        import_map_action.triggered.connect(self._import_mapping_csv)
        save_sess_action = QAction("Save Session…", self)
        save_sess_action.triggered.connect(self._save_session)
        load_sess_action = QAction("Load Session…", self)
        load_sess_action.triggered.connect(self._load_session)

        file_menu.addAction(open_action)
        file_menu.addAction(import_map_action)
        file_menu.addSeparator()
        file_menu.addAction(save_sess_action)
        file_menu.addAction(load_sess_action)

        self.toolbar = self.addToolBar("Main")
        self.toolbar.addAction(open_action)
        self.toolbar.addAction(import_map_action)

        help_menu = self.menuBar().addMenu("Help")
        quickstart_action = QAction("Quick Start", self)
        quickstart_action.triggered.connect(self._show_quick_start)
        help_menu.addAction(quickstart_action)

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

        # Right pane: tabs + plot
        right_widget = QWidget(splitter)
        right_layout = QVBoxLayout(right_widget)

        # Tabs for new manual Selections (A/B) and legacy Metric Builder
        self.tabs = QTabWidget(right_widget)
        self.selection_panel = SelectionPanel(self.tabs)
        self.calc_builder = CalcBuilderWidget(self.tabs)
        self.tabs.addTab(self.selection_panel, "Selections")
        self.tabs.addTab(self.calc_builder, "Metric Builder")
        self.tabs.setCurrentIndex(0)

        # Plot controls below tabs
        self.plot_config = PlotConfigWidget(right_widget)
        self.trendline_form = TrendlineForm(right_widget)
        self.canvas = PlotCanvas()
        self.toolbar_canvas = NavigationToolbar2QT(self.canvas, right_widget)

        right_layout.addWidget(self.tabs)
        right_layout.addWidget(self.plot_config)
        right_layout.addWidget(self.trendline_form)
        right_layout.addWidget(self.toolbar_canvas)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 960])  # give the right pane enough initial real estate

        central_layout.addWidget(splitter)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar(self))

    def _connect_signals(self) -> None:
        self.file_list.tagChanged.connect(self._on_tag_changed)
        self.file_list.selectionChanged.connect(self._on_file_selected)
        self.file_list.xChanged.connect(self._on_x_changed)
        self.data_table.cellPicked.connect(self._on_cell_picked)
        self.calc_builder.metricComputed.connect(self._on_metric_computed)
        self.selection_panel.autopopulateRequested.connect(self._on_autopopulate_requested)
        self.selection_panel.metricsUpdated.connect(self._on_selection_metrics_updated)
        self.plot_config.plotRequested.connect(self._on_plot_requested)
        self.plot_config.exportMetricsRequested.connect(self._export_metrics)
        self.plot_config.exportPlotRequested.connect(self._export_plot)
        self.plot_config.exportXYRequested.connect(self._export_xy)
        self.plot_config.exportGroupStatsRequested.connect(self._export_group_stats)
        self.trendline_form.fitRequested.connect(self._on_fit_requested)
        self.trendline_form.literatureOverlayRequested.connect(self._on_literature_overlay)
        self.trendline_form.intersectionsRequested.connect(self._on_intersections_requested)
        self.trendline_form.exportIntersectionsRequested.connect(
            self._export_intersections
        )
        self.trendline_form.exportFitsRequested.connect(self._export_fit_params)
        self.trendline_form.exportResidualsRequested.connect(self._export_residuals)

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
        if self._loader_thread and self._loader_thread.isRunning():
            QMessageBox.information(
                self,
                "Load CSVs",
                "A CSV load is already in progress. Please wait for it to finish.",
            )
            return

        self._load_failed = False
        self.statusBar().showMessage("Loading CSVs…")
        worker = CsvLoaderWorker(paths)
        thread = QThread(self)
        self._loader_thread = thread
        self._loader_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_csvs_loaded)
        worker.tablesLoaded.connect(self._on_tables_loaded)
        worker.error.connect(self._on_csv_load_error)
        worker.finished.connect(lambda _: thread.quit())
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: setattr(self, "_loader_thread", None))
        thread.finished.connect(lambda: setattr(self, "_loader_worker", None))
        thread.start()

    @pyqtSlot(pd.DataFrame)
    def _on_csvs_loaded(self, df: pd.DataFrame) -> None:
        if df.empty:
            if not self._load_failed:
                QMessageBox.warning(
                    self, "Load CSVs", "No data found in selected files."
                )
            self.statusBar().clearMessage()
            self._load_failed = False
            return
        self.session.set_raw_data(df)
        files = df["file"].dropna().unique().tolist()
        self.file_list.set_files(
            files, self.session.file_to_tag, self.session.x_mapping or {}
        )
        self.calc_builder.set_data(df, self.session.file_to_tag)
        self.selection_panel.set_context(self.session.file_to_tag)
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
        self._load_failed = False

    def _on_tables_loaded(self, tables: dict) -> None:
        self.session.set_raw_tables(tables or {})
        self.selection_panel.set_context(self.session.file_to_tag)
        files = self.file_list.selected_files
        if not files and not self.session.raw_df.empty and "file" in self.session.raw_df.columns:
            try:
                files = (
                    self.session.raw_df["file"].dropna().astype(str).unique().tolist()
                )
            except Exception:
                files = []
        if files:
            self._show_grid_for_file(files[0])

    def _on_csv_load_error(self, message: str) -> None:  # pragma: no cover - GUI warning
        self._load_failed = True
        QMessageBox.warning(self, "Load CSVs", message)
        self.statusBar().clearMessage()

    # ------------------------------------------------------------------ Session save/load
    def _save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "Raman Analyzer Session (*.json)"
        )
        if not path:
            return
        try:
            # Persist the visible widget state even if the user hasn't clicked Plot yet.
            cfg = self.plot_config.current_config()
            self.session.plot_config = cfg if cfg else None
            try:
                self.session.selection_state = self.selection_panel.get_state()
            except Exception:
                self.session.selection_state = None
            save_session(path, self.session)
        except Exception as exc:  # pragma: no cover - defensive feedback
            QMessageBox.warning(self, "Save Session", f"Failed to save session:\n{exc}")
            return
        self.statusBar().showMessage(f"Session saved to {path}", 5000)

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "Raman Analyzer Session (*.json)"
        )
        if not path:
            return
        try:
            sess = load_session(path)
        except Exception as exc:  # pragma: no cover - defensive feedback
            QMessageBox.warning(self, "Load Session", f"Failed to load session:\n{exc}")
            return

        self.selection_panel.blockSignals(True)
        try:
            self.selection_panel.apply_state({})
        finally:
            self.selection_panel.blockSignals(False)

        self.session = sess
        self.current_plot_config = None
        self.selection_panel.set_context(self.session.file_to_tag)
        df = self.session.raw_df if self.session.raw_df is not None else pd.DataFrame()
        if df.empty:
            QMessageBox.information(
                self,
                "Load Session",
                "Session loaded, but no raw data found.",
            )
            self.file_list.set_files([], {}, {})
            self.data_table.set_dataframe(pd.DataFrame())
            self.calc_builder.set_data(pd.DataFrame(), {})
            self.selection_panel.set_context({})
            self.calc_builder.set_available_attributes(
                ["area", "height", "fwhm", "center", "area_pct"]
            )
            self.plot_config.set_metrics([])
            self.current_plot_data = None
            self.canvas.axes.cla()
            self.canvas.draw()
            self.trendline_form.set_fit_summary(self._format_fit_summary(None))
            self.trendline_form.set_intersections([])
            self.statusBar().showMessage(f"Session loaded from {path}", 5000)
            return

        if "file" not in df.columns:
            QMessageBox.warning(
                self,
                "Load Session",
                "Session data is missing the 'file' column and cannot be displayed.",
            )
            self.file_list.set_files([], {}, {})
            self.data_table.set_dataframe(pd.DataFrame())
            self.calc_builder.set_data(df, {})
            self.selection_panel.set_context({})
            self.calc_builder.set_available_attributes(
                ["area", "height", "fwhm", "center", "area_pct"]
            )
            self.plot_config.set_metrics([])
            self.current_plot_data = None
            self.canvas.axes.cla()
            self.canvas.draw()
            self.trendline_form.set_fit_summary(self._format_fit_summary(None))
            self.trendline_form.set_intersections([])
            self.statusBar().showMessage(f"Session loaded from {path}", 5000)
            return

        files = df["file"].dropna().unique().tolist()
        if not files:
            QMessageBox.information(
                self,
                "Load Session",
                "Session loaded, but no files were found in the data.",
            )
            self.file_list.set_files([], self.session.file_to_tag, self.session.x_mapping or {})
            self.data_table.set_dataframe(pd.DataFrame())
            self.calc_builder.set_data(df, self.session.file_to_tag)
            available_attrs = [
                col
                for col in ["area", "height", "fwhm", "center", "area_pct"]
                if col in df.columns
            ] or ["area", "height", "fwhm", "center", "area_pct"]
            self.calc_builder.set_available_attributes(available_attrs)
            self.plot_config.set_metrics([])
            self.current_plot_data = None
            self.canvas.axes.cla()
            self.canvas.draw()
            self.trendline_form.set_fit_summary(self._format_fit_summary(self.session.data_fit))
            self.trendline_form.set_intersections(self.session.intersections)
            self.statusBar().showMessage(f"Session loaded from {path}", 5000)
            return

        self.file_list.set_files(
            files, self.session.file_to_tag, self.session.x_mapping or {}
        )
        self.calc_builder.set_data(df, self.session.file_to_tag)
        self.selection_panel.set_context(self.session.file_to_tag)
        available_attrs = [
            col
            for col in ["area", "height", "fwhm", "center", "area_pct"]
            if col in df.columns
        ] or ["area", "height", "fwhm", "center", "area_pct"]
        self.calc_builder.set_available_attributes(available_attrs)
        self._update_plot_metrics()
        self._on_file_selected(files[:1])
        self.trendline_form.set_fit_summary(self._format_fit_summary(self.session.data_fit))
        self.trendline_form.set_intersections(self.session.intersections)

        try:
            sel_state = getattr(self.session, "selection_state", None)
            if isinstance(sel_state, dict) and sel_state:
                self.selection_panel.apply_state(sel_state)
        except Exception:
            pass

        restored_cfg = getattr(self.session, "plot_config", None)
        if isinstance(restored_cfg, dict):
            restored_cfg = dict(restored_cfg)
            self.plot_config.apply_config(restored_cfg)
            self.current_plot_config = restored_cfg
            available_metrics = [
                col for col in self.session.results_df.columns if col not in {"file", "tag"}
            ]
            y_metric = restored_cfg.get("y_axis")
            if isinstance(y_metric, str) and y_metric in available_metrics:
                self._on_plot_requested(restored_cfg)
            else:
                self.canvas.axes.cla()
                self.canvas.draw()
                self.current_plot_data = None
        else:
            self.canvas.axes.cla()
            self.canvas.draw()
            self.current_plot_data = None
        self.statusBar().showMessage(f"Session loaded from {path}", 5000)

    def _show_quick_start(self) -> None:
        QMessageBox.information(
            self,
            "Quick Start",
            "1) Load CSVs of peak fits.\n"
            "2) Assign Tags/X in the left table or import a mapping CSV.\n"
            "3) In 'Metric Builder', choose a mode (e.g., Ratio or Normalized Area) and Compute.\n"
            "4) Configure the plot (X/Y, type, error bars) and click Plot.\n"
            "5) (Optional) Fit data trendline, overlay literature, compute intersections.\n"
            "6) Export metrics/plot/residuals/intersections from the buttons.",
        )

    # ------------------------------------------------------------------ File interactions
    def _on_file_selected(self, files: List[str]) -> None:
        if not files:
            self._current_grid_file = None
            self.data_table.set_dataframe(pd.DataFrame())
            return
        self._show_grid_for_file(files[0])

    def _show_grid_for_file(self, file_id: str) -> None:
        self._current_grid_file = file_id
        table = self.session.get_raw_table(file_id)
        if table is None or table.empty:
            if not self.session.raw_df.empty and "file" in self.session.raw_df.columns:
                try:
                    subset = self.session.raw_df[
                        self.session.raw_df["file"].astype(str) == str(file_id)
                    ]
                except Exception:
                    subset = pd.DataFrame()
            else:
                subset = pd.DataFrame()
            self.data_table.set_dataframe_for_file(file_id, subset)
            if subset.empty:
                self.statusBar().showMessage(f"No raw grid for '{file_id}'", 4000)
            else:
                self.statusBar().showMessage(
                    f"No raw grid for '{file_id}', showing tidy rows", 4000
                )
            return
        self.data_table.set_dataframe_for_file(file_id, table)
        self.statusBar().showMessage(f"Showing grid for '{file_id}'", 2000)

    def _on_cell_picked(self, file_id: str, row1: int, col1: int, value: object) -> None:
        try:
            numeric = float(value)
        except Exception:
            return
        tag = self.session.file_to_tag.get(str(file_id), "")
        self.selection_panel.add_pick(
            str(file_id),
            int(row1),
            int(col1),
            float(numeric),
            target=None,
            tag=tag,
        )
        self.statusBar().showMessage(
            f"Picked {numeric:.6g} from {file_id} ({row1},{col1})",
            2500,
        )

    def _on_autopopulate_requested(self, target: str, row1: int, col1: int, scope: str) -> None:
        if scope == "Selected":
            files = list(self.file_list.selected_files)
        else:
            try:
                if not self.session.raw_df.empty and "file" in self.session.raw_df.columns:
                    files = (
                        self.session.raw_df["file"].dropna().astype(str).unique().tolist()
                    )
                else:
                    files = []
            except Exception:
                files = []
        if not files:
            self.statusBar().showMessage("No files to populate.", 3000)
            return
        added = 0
        r0 = int(row1) - 1
        c0 = int(col1) - 1
        for file_id in files:
            table = self.session.get_raw_table(file_id)
            if table is None or table.empty:
                continue
            if r0 < 0 or c0 < 0:
                continue
            try:
                cell = table.iat[r0, c0]
            except Exception:
                continue
            try:
                value = float(cell)
            except Exception:
                continue
            tag = self.session.file_to_tag.get(str(file_id), "")
            self.selection_panel.add_pick(
                str(file_id),
                int(row1),
                int(col1),
                float(value),
                target=target,
                tag=tag,
            )
            added += 1
        self.statusBar().showMessage(
            f"Auto-populated {added} value(s) to {target}",
            4000,
        )

    def _on_selection_metrics_updated(self, a_payload: object, b_payload: object) -> None:
        try:
            a_name, a_df = a_payload  # type: ignore[misc]
            b_name, b_df = b_payload  # type: ignore[misc]
        except Exception:
            return
        if isinstance(a_df, pd.DataFrame):
            a_values = a_df[["file", "value"]] if not a_df.empty else pd.DataFrame(
                columns=["file", "value"]
            )
            self.session.update_metric(str(a_name), a_values)
        if isinstance(b_df, pd.DataFrame):
            b_values = b_df[["file", "value"]] if not b_df.empty else pd.DataFrame(
                columns=["file", "value"]
            )
            self.session.update_metric(str(b_name), b_values)
        self._update_plot_metrics()
        try:
            self.session.selection_state = self.selection_panel.get_state()
        except Exception:
            self.session.selection_state = None

    def _on_tag_changed(self, file_id: str, tag: str) -> None:
        self.session.set_tag(file_id, tag)
        self.calc_builder.set_data(self.session.raw_df, self.session.file_to_tag)
        self.selection_panel.set_context(self.session.file_to_tag)
        self._update_plot_metrics()

    def _on_x_changed(self, file_id: str, x_value: object) -> None:
        mapping = dict(self.session.x_mapping or {})
        if x_value is None:
            mapping.pop(file_id, None)
        else:
            mapping[file_id] = float(x_value)
        self.session.update_x_mapping(mapping)
        if not self.session.raw_df.empty:
            files = self.session.raw_df["file"].dropna().unique().tolist()
            self.file_list.set_files(
                files, self.session.file_to_tag, self.session.x_mapping or {}
            )
        if self.current_plot_config:
            self._on_plot_requested(self.current_plot_config)

    # ------------------------------------------------------------------ Metrics handling
    def _on_metric_computed(self, metric_name: str, long_df: pd.DataFrame) -> None:
        per_file = long_df[["file", "value"]]
        self.session.update_metric(metric_name, per_file)
        self._update_plot_metrics()
        self.statusBar().showMessage(f"Metric '{metric_name}' computed", 5000)

    def _update_plot_metrics(self) -> None:
        """Refresh the plot metric dropdowns from results_df with friendly ordering.

        We pin Selection A/B at the top (if present), then list all other metrics.
        """
        df = self.session.results_df
        if df is None or df.empty:
            self.plot_config.set_metrics([])
            return
        cols = [col for col in df.columns if col not in {"file", "tag"}]
        preferred = ["Selection A", "Selection B"]
        pinned = [col for col in preferred if col in cols]
        rest = [col for col in cols if col not in pinned]
        self.plot_config.set_metrics(pinned + rest)

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

        if x_axis == "Group (categorical)":
            plot_df = df[["file", "tag", y_metric]].dropna()
            if plot_df.empty:
                return
            categories = pd.Categorical(plot_df["tag"].fillna("(untagged)"))
            numeric_x = categories.codes.astype(float)
            plot_df = plot_df.assign(tag=categories, x=numeric_x, y=plot_df[y_metric])
            if plot_type == "Line":
                draw_line(self.canvas.axes, plot_df, "x", "y", hue="tag")
            elif plot_type == "Scatter":
                draw_scatter(
                    self.canvas.axes,
                    plot_df,
                    "x",
                    "y",
                    hue="tag",
                    jitter=config.get("jitter", False),
                )
            self.canvas.axes.set_xticks(range(len(categories.categories)))
            self.canvas.axes.set_xticklabels(categories.categories)
            self.canvas.axes.set_xlabel("Group")
            self.canvas.axes.set_ylabel(y_metric)
        elif x_axis == "Custom X (per file)":
            if not self.session.x_mapping:
                QMessageBox.information(
                    self, "Plot", "Please enter X values in the File list."
                )
                return
            tmp = df[["file", "tag", y_metric]].copy()
            tmp["x"] = tmp["file"].map(self.session.x_mapping).astype(float)
            plot_df = tmp.dropna(subset=["x", y_metric]).rename(
                columns={y_metric: "y"}
            )
            # Scatter/Line with computed numeric X
            if plot_df.empty:
                return
            if plot_type == "Line":
                draw_line(self.canvas.axes, plot_df, "x", "y", hue="tag")
            elif plot_type == "Scatter":
                draw_scatter(
                    self.canvas.axes,
                    plot_df,
                    "x",
                    "y",
                    hue="tag",
                    jitter=config.get("jitter", False),
                )
            self.canvas.axes.set_xlabel("X")
            self.canvas.axes.set_ylabel(y_metric)
        else:
            # X-axis taken from the dataset (may be categorical or numeric)
            if x_axis not in df.columns:
                return
            plot_df = df[["file", "tag", x_axis, y_metric]].dropna()
            if plot_df.empty:
                return
            plot_df = plot_df.rename(columns={x_axis: "x", y_metric: "y"})
            if plot_type == "Line":
                draw_line(self.canvas.axes, plot_df, "x", "y", hue="tag")
            elif plot_type == "Scatter":
                draw_scatter(
                    self.canvas.axes,
                    plot_df,
                    "x",
                    "y",
                    hue="tag",
                    jitter=config.get("jitter", False),
                )
            self.canvas.axes.set_xlabel(x_axis)
            self.canvas.axes.set_ylabel(y_metric)
        self.current_plot_data = plot_df[["file", "tag", "x", "y"]]

        # --- Box/Violin plots (categorical by tag) ---
        if plot_type in ("Box", "Violin"):
            # For box/violin we ignore numeric X and visualize distributions per tag (group).
            # Build y-value lists per tag from the current data scope.
            base = self.current_plot_data if self.current_plot_data is not None else None
            if base is None or base.empty:
                return
            # Use the latest y metric values per group
            grouped = (
                base.dropna(subset=["y"])
                .groupby("tag", dropna=False)["y"]
                .apply(lambda s: s.astype(float).to_list())
            )
            if not len(grouped):
                return
            # Keep display order stable by tag name
            labels = [str(t) for t in grouped.index.tolist()]
            data = list(grouped.values)
            self._draw_box_or_violin(kind=plot_type, labels=labels, data=data)
            self.canvas.axes.set_xlabel("Group")
            self.canvas.axes.set_ylabel(y_metric)
            self.current_plot_config = config
            self.canvas.draw()
            self._overlay_trendlines()
            return

        # Add uncertainty after drawing the primary glyphs (Scatter/Line), before overlaying fits.
        if plot_type == "Scatter":
            self._add_error_bars(config.get("error_mode", "None"))
        elif plot_type == "Line":
            self._shade_line_uncertainty(config.get("error_mode", "None"))

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

    def _add_error_bars(self, mode: str) -> None:
        if self.current_plot_data is None or self.current_plot_data.empty:
            return
        if mode == "None":
            return

        grouped = compute_error_table(self.current_plot_data, mode=mode)
        if grouped.empty:
            return

        mask = (grouped["yerr"] > 0) & np.isfinite(grouped["yerr"])
        if not mask.any():
            return

        self.canvas.axes.errorbar(
            grouped.loc[mask, "x"],
            grouped.loc[mask, "mean"],
            yerr=grouped.loc[mask, "yerr"],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
            zorder=3,
        )

    # -------- Box/Violin helpers --------
    def _draw_box_or_violin(self, kind: str, labels: list, data: list) -> None:
        """
        Render a box or violin plot over per-group distributions.

        Parameters
        ----------
        kind : {"Box", "Violin"}
            Plot type to render.
        labels : list[str]
            Group labels for the x-axis.
        data : list[list[float]]
            Each entry is the list of y-values for the corresponding group.
        """
        ax = self.canvas.axes
        ax.cla()
        # Positions are 1..N so we can set xticks nicely
        positions = list(range(1, len(labels) + 1))
        if kind == "Box":
            bp = ax.boxplot(
                data,
                positions=positions,
                showmeans=True,
                meanline=False,
                vert=True,
                patch_artist=True,
            )
            # Subtle alpha fill so distributions are visible; color per default cycle
            for patch in bp.get("boxes", []):
                patch.set_alpha(0.25)
        else:  # "Violin"
            vp = ax.violinplot(
                data,
                positions=positions,
                showmeans=True,
                showextrema=True,
                showmedians=False,
                vert=True,
            )
            # Make the violins slightly translucent
            for body in vp.get("bodies", []):
                body.set_alpha(0.25)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)

    def _overlay_trendlines(self) -> None:
        if self.current_plot_data is None or self.current_plot_data.empty:
            return
        x_min = float(self.current_plot_data["x"].min())
        x_max = float(self.current_plot_data["x"].max())
        overlay_data_fit(self.canvas.axes, self.session.data_fit, (x_min, x_max))
        overlay_literature(self.canvas.axes, self.session.literature_fit, (x_min, x_max))
        mark_intersections(self.canvas.axes, self.session.intersections)
        handles, labels = self.canvas.axes.get_legend_handles_labels()
        if labels:
            unique: dict[str, object] = {}
            for handle, label in zip(handles, labels):
                if label not in unique:
                    unique[label] = handle
            self.canvas.axes.legend(unique.values(), unique.keys())
        self.canvas.draw()

    # -------- Line-plot uncertainty shading --------
    def _shade_line_uncertainty(self, mode: str) -> None:
        """For line plots, draw a shaded band (SD/SEM/95% CI) around the mean line per tag."""
        if (
            self.current_plot_data is None
            or self.current_plot_data.empty
            or mode == "None"
        ):
            return
        grouped = compute_error_table(self.current_plot_data, mode=mode)
        if grouped.empty:
            return

        # Map plotted group lines to their colors so the band matches.
        # We call this before overlaying fits, so the only lines should be the groups.
        line_colors: dict[str, str] = {}
        for line in self.canvas.axes.get_lines():
            label = line.get_label()
            if label:
                try:
                    color = line.get_color()
                except Exception:
                    color = None
                if color:
                    line_colors[str(label)] = color

        for tag, g in grouped.groupby("tag", dropna=False):
            # Only shade where we have a positive, finite uncertainty and finite mean.
            mask = (
                np.isfinite(g["mean"])
                & np.isfinite(g["yerr"])
                & (g["yerr"] > 0)
            )
            if not mask.any():
                continue
            sg = g.loc[mask].sort_values("x")
            x = sg["x"].to_numpy(dtype=float, copy=False)
            m = sg["mean"].to_numpy(dtype=float, copy=False)
            e = sg["yerr"].to_numpy(dtype=float, copy=False)
            lower = m - e
            upper = m + e
            color = line_colors.get(str(tag), None)
            self.canvas.axes.fill_between(
                x, lower, upper, alpha=0.2, linewidth=0, color=color, zorder=1
            )

    # ------------------------------------------------------------------ Trendlines
    def _format_fit_summary(self, fit: Optional[dict]) -> str:
        if not fit:
            return "No fit computed"
        model = fit.get("model")
        coeffs = fit.get("coeffs", ())
        r2 = fit.get("r2")
        if model == "linear" and len(coeffs) >= 2:
            return (
                f"Linear fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}"
                + (f" (R^2={r2:.3f})" if r2 is not None else "")
            )
        if model == "quadratic" and len(coeffs) >= 3:
            a, b, c = coeffs[:3]
            return (
                f"Quadratic fit: y={a:.3f}x²+{b:.3f}x+{c:.3f}"
                + (f" (R^2={r2:.3f})" if r2 is not None else "")
            )
        if model == "power" and len(coeffs) >= 2:
            return (
                f"Power fit: y={coeffs[0]:.3f}x^{coeffs[1]:.3f}"
                + (f" (R^2={r2:.3f})" if r2 is not None else "")
            )
        return "Fit loaded"

    def _on_fit_requested(self, model: str) -> None:
        if self.current_plot_data is None or self.current_plot_data.empty:
            QMessageBox.information(
                self, "Trendline", "Plot data is required before fitting."
            )
            return
        x = self.current_plot_data["x"].to_numpy(dtype=float)
        y = self.current_plot_data["y"].to_numpy(dtype=float)
        if len(x) < 2:
            QMessageBox.warning(self, "Trendline", "Not enough data points to fit.")
            return
        model = model.lower()
        if model == "quadratic":
            fit = fit_quadratic(x, y)
        elif model == "power":
            try:
                fit = fit_power(x, y)
            except ValueError:
                QMessageBox.warning(
                    self, "Trendline", "Power fit requires positive x and y values."
                )
                return
        else:
            fit = fit_linear(x, y)
        self.session.set_data_fit(fit)
        summary = self._format_fit_summary(fit)
        self.trendline_form.set_fit_summary(summary)
        if self.current_plot_config:
            self._on_plot_requested(self.current_plot_config)

    def _on_literature_overlay(self, payload: dict) -> None:
        model = payload.get("model")
        coeffs = payload.get("coeffs", ())
        if model not in {"linear", "quadratic", "power"}:
            QMessageBox.warning(self, "Trendline", "Unsupported literature model.")
            return
        if (model == "linear" and len(coeffs) != 2) or (
            model == "quadratic" and len(coeffs) != 3
        ) or (model == "power" and len(coeffs) != 2):
            QMessageBox.warning(self, "Trendline", "Incomplete literature coefficients.")
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
        data_model = self.session.data_fit.get("model")
        lit_model = self.session.literature_fit.get("model")
        data_coeffs = self.session.data_fit.get("coeffs", ())
        lit_coeffs = self.session.literature_fit.get("coeffs", ())

        if data_model == "linear" and lit_model == "linear":
            points = intersections_linear_linear(*data_coeffs, *lit_coeffs)
        elif data_model == "quadratic" and lit_model == "linear":
            a, b, c = data_coeffs
            m, b2 = lit_coeffs
            points = intersections_poly_linear(a, b, c, m, b2)
        elif data_model == "linear" and lit_model == "quadratic":
            a, b, c = lit_coeffs
            m, b2 = data_coeffs
            points = intersections_poly_linear(a, b, c, m, b2)
        else:
            if self.current_plot_data is None or self.current_plot_data.empty:
                points = []
            else:
                x_min = float(self.current_plot_data["x"].min())
                x_max = float(self.current_plot_data["x"].max())

                def _wrap(model_name: str, coeffs: tuple[float, ...]):
                    if model_name == "linear":
                        m, b = coeffs
                        return lambda x: eval_linear(np.array([x]), m, b)[0]
                    if model_name == "quadratic":
                        a_, b_, c_ = coeffs
                        return lambda x: eval_quadratic(np.array([x]), a_, b_, c_)[0]
                    A_, B_ = coeffs
                    return lambda x: eval_power(np.array([x]), A_, B_)[0]

                f1 = _wrap(data_model, data_coeffs)
                f2 = _wrap(lit_model, lit_coeffs)
                points = intersections_numeric(f1, f2, x_min, x_max, steps=200)
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
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        self.canvas.figure.savefig(path, dpi=300)
        self.statusBar().showMessage(f"Plot saved to {path}", 5000)

    def _export_xy(self) -> None:
        if self.current_plot_data is None or self.current_plot_data.empty:
            QMessageBox.information(self, "Export XY", "No XY data to export.")
            return
        x_label = "x"
        y_label = "y"
        if self.current_plot_config:
            x_label = self.current_plot_config.get("x_axis", x_label)
            y_label = self.current_plot_config.get("y_axis", y_label)
        df = self.current_plot_data.copy()
        df = df.rename(columns={"x": x_label or "x", "y": y_label or "y"})
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export current XY CSV",
            "plot_xy.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        df.to_csv(path, index=False)
        self.statusBar().showMessage(f"XY exported to {path}", 5000)

    def _export_group_stats(self) -> None:
        """Export grouped stats with error columns (SD/SEM/95% CI)."""
        if self.current_plot_data is None or self.current_plot_data.empty:
            QMessageBox.information(
                self,
                "Export Group Stats",
                "There is no plotted data to summarize. Please plot a metric first.",
            )
            return

        mode = "None"
        if isinstance(self.current_plot_config, dict):
            mode = self.current_plot_config.get("error_mode", "None") or "None"

        table = compute_error_table(self.current_plot_data, mode=mode)
        if table.empty:
            QMessageBox.information(
                self,
                "Export Group Stats",
                "Could not compute grouped statistics for the current view.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Group Stats CSV",
            "",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            table.to_csv(path, index=False)
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Export Group Stats", f"Failed to save CSV:\n{exc}")
            return
        self.statusBar().showMessage(f"Group stats saved to {path}", 5000)

    def _export_residuals(self) -> None:
        """Export a CSV combining current XY, model predictions, and residuals."""
        if self.current_plot_data is None or self.current_plot_data.empty:
            QMessageBox.information(self, "Export Residuals", "No plot data available.")
            return
        if not self.session.data_fit:
            QMessageBox.information(
                self, "Export Residuals", "No data fit to compare against."
            )
            return

        fit = self.session.data_fit
        model = str(fit.get("model", "")).lower()
        coeffs = tuple(fit.get("coeffs", ()))
        df = self.current_plot_data.copy()
        x = df["x"].astype(float).to_numpy()

        if model == "linear" and len(coeffs) >= 2:
            y_hat = coeffs[0] * x + coeffs[1]
        elif model == "quadratic" and len(coeffs) >= 3:
            y_hat = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
        elif model == "power" and len(coeffs) >= 2:
            y_hat = np.where(x > 0, coeffs[0] * np.power(x, coeffs[1]), np.nan)
        else:
            QMessageBox.warning(
                self, "Export Residuals", "Unsupported or incomplete fit model."
            )
            return

        # Compose output: XY + prediction + residuals
        df_out = df.copy()
        df_out["y_fit"] = y_hat
        df_out["residual"] = df_out["y"].astype(float) - df_out["y_fit"]

        # Use friendly headers aligned with the current plot configuration.
        x_label = "x"
        y_label = "y"
        if self.current_plot_config:
            x_label = self.current_plot_config.get("x_axis", x_label) or x_label
            y_label = self.current_plot_config.get("y_axis", y_label) or y_label
        save_df = df_out.rename(columns={"x": x_label, "y": y_label})

        default_name = f"data_fit_residuals_{y_label}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Data + Residuals CSV", default_name, "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            save_df.to_csv(path, index=False)
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.warning(self, "Export Residuals", f"Failed to save CSV:\n{exc}")
            return
        self.statusBar().showMessage(f"Data + residuals exported to {path}", 5000)

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

    def _export_fit_params(self) -> None:
        if not self.session.data_fit and not self.session.literature_fit:
            QMessageBox.information(
                self, "Export Fit Params", "No fit parameters to export."
            )
            return
        rows = []
        if self.session.data_fit:
            data_fit = self.session.data_fit
            coeffs = list(data_fit.get("coeffs", ()))
            rows.append(
                {
                    "role": "data",
                    "model": data_fit.get("model", ""),
                    "r2": data_fit.get("r2", ""),
                    "coeff1": coeffs[0] if len(coeffs) > 0 else "",
                    "coeff2": coeffs[1] if len(coeffs) > 1 else "",
                    "coeff3": coeffs[2] if len(coeffs) > 2 else "",
                }
            )
        if self.session.literature_fit:
            lit_fit = self.session.literature_fit
            coeffs = list(lit_fit.get("coeffs", ()))
            rows.append(
                {
                    "role": "literature",
                    "model": lit_fit.get("model", ""),
                    "r2": "",
                    "coeff1": coeffs[0] if len(coeffs) > 0 else "",
                    "coeff2": coeffs[1] if len(coeffs) > 1 else "",
                    "coeff3": coeffs[2] if len(coeffs) > 2 else "",
                }
            )
        df = pd.DataFrame(rows, columns=["role", "model", "r2", "coeff1", "coeff2", "coeff3"])
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export fit parameters CSV",
            "fit_params.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        df.to_csv(path, index=False)
        self.statusBar().showMessage(f"Fit params exported to {path}", 5000)

    def _import_mapping_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Tags/X mapping CSV",
            "",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            mapping_df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - user feedback path
            QMessageBox.warning(self, "Import mapping", f"Failed to read CSV:\n{exc}")
            return
        if "file" not in mapping_df.columns:
            QMessageBox.warning(self, "Import mapping", "CSV must contain a 'file' column.")
            return
        if "tag" in mapping_df.columns:
            for _, row in mapping_df[["file", "tag"]].dropna(subset=["file"]).iterrows():
                self.session.set_tag(str(row["file"]), str(row["tag"]))
        if "x" in mapping_df.columns:
            mapping = dict(self.session.x_mapping or {})
            for _, row in mapping_df[["file", "x"]].dropna(subset=["file"]).iterrows():
                try:
                    mapping[str(row["file"])] = float(row["x"])
                except (TypeError, ValueError):
                    continue
            self.session.update_x_mapping(mapping)
        if not self.session.raw_df.empty:
            files = self.session.raw_df["file"].dropna().unique().tolist()
            self.file_list.set_files(
                files,
                self.session.file_to_tag,
                self.session.x_mapping or {},
            )
        self._update_plot_metrics()
        if self.current_plot_config:
            self._on_plot_requested(self.current_plot_config)
        self.statusBar().showMessage("Mapping CSV imported", 5000)
