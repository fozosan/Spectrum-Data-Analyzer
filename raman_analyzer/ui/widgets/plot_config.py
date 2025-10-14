"""Plot configuration controls."""
from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PlotConfigWidget(QWidget):
    """Collect plotting options from the user."""

    plotRequested = pyqtSignal(dict)
    exportPlotRequested = pyqtSignal()
    exportMetricsRequested = pyqtSignal()
    exportGroupStatsRequested = pyqtSignal()
    exportXYRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metrics: List[str] = []
        self.x_combo = QComboBox(self)
        self.y_combo = QComboBox(self)
        self.plot_type_combo = QComboBox(self)
        self.plot_type_combo.addItems(["Scatter", "Line", "Box", "Violin"])
        self.error_mode_combo = QComboBox(self)
        self.error_mode_combo.addItems(["None", "SD", "SEM", "95% CI"])
        self.error_mode_combo.setToolTip(
            "Applies to Scatter: SD (sample std), SEM (std/√n), or 95% CI using t-critical values.\n"
            "Groups with fewer than two points omit error bars."
        )

        self.xmin_edit = QLineEdit(self)
        self.xmax_edit = QLineEdit(self)
        self.ymin_edit = QLineEdit(self)
        self.ymax_edit = QLineEdit(self)

        for edit in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            edit.setPlaceholderText("Auto")

        form = QFormLayout()
        form.addRow("X axis", self.x_combo)
        form.addRow("Y axis", self.y_combo)
        form.addRow("Plot type", self.plot_type_combo)
        form.addRow("Error bars", self.error_mode_combo)

        axis_group = QGroupBox("Axis limits", self)
        axis_layout = QFormLayout(axis_group)
        axis_layout.addRow("X min", self.xmin_edit)
        axis_layout.addRow("X max", self.xmax_edit)
        axis_layout.addRow("Y min", self.ymin_edit)
        axis_layout.addRow("Y max", self.ymax_edit)

        button_layout = QHBoxLayout()
        self.plot_button = QPushButton("Plot", self)
        self.export_plot_button = QPushButton("Export Plot", self)
        self.export_metrics_button = QPushButton("Export Metrics CSV", self)
        self.export_group_stats_button = QPushButton("Export Group Stats CSV", self)
        self.export_xy_button = QPushButton("Export Current XY CSV", self)
        button_layout.addWidget(self.plot_button)
        button_layout.addWidget(self.export_plot_button)
        button_layout.addWidget(self.export_metrics_button)
        button_layout.addWidget(self.export_group_stats_button)
        button_layout.addWidget(self.export_xy_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(axis_group)
        layout.addLayout(button_layout)
        layout.addStretch(1)

        self.plot_type_combo.currentTextChanged.connect(self._toggle_error_mode_enabled)
        self._toggle_error_mode_enabled(self.plot_type_combo.currentText())

        self.plot_button.clicked.connect(self._emit_plot)
        self.export_plot_button.clicked.connect(self.exportPlotRequested.emit)
        self.export_metrics_button.clicked.connect(self.exportMetricsRequested.emit)
        self.export_group_stats_button.clicked.connect(
            self.exportGroupStatsRequested.emit
        )
        self.export_xy_button.clicked.connect(self.exportXYRequested.emit)

    def set_metrics(self, metrics: List[str]) -> None:
        """Update the metric dropdowns."""

        self.metrics = metrics
        current_y = self.y_combo.currentText()
        current_x = self.x_combo.currentText()

        self.y_combo.clear()
        self.y_combo.addItems(metrics)

        self.x_combo.clear()
        self.x_combo.addItem("Group (categorical)")
        self.x_combo.addItem("Custom X (per file)")
        self.x_combo.addItems(metrics)

        if current_y in metrics:
            self.y_combo.setCurrentText(current_y)
        if current_x and (
            current_x == "Group (categorical)"
            or current_x == "Custom X (per file)"
            or current_x in metrics
        ):
            self.x_combo.setCurrentText(current_x)

    def current_config(self) -> dict:
        """Return the currently visible plot configuration.

        The structure mirrors the dict emitted by :meth:`_emit_plot`, which
        allows callers to persist UI state even if the user hasn't clicked
        "Plot" yet.
        """

        return {
            "x_axis": self.x_combo.currentText(),
            "y_axis": self.y_combo.currentText(),
            "plot_type": self.plot_type_combo.currentText(),
            "error_mode": self.error_mode_combo.currentText(),
            "x_limits": (
                self._parse_limit(self.xmin_edit),
                self._parse_limit(self.xmax_edit),
            ),
            "y_limits": (
                self._parse_limit(self.ymin_edit),
                self._parse_limit(self.ymax_edit),
            ),
        }

    def apply_config(self, cfg: dict) -> None:
        """Restore UI controls from a previously saved configuration."""

        if not isinstance(cfg, dict):
            return

        def _has_value(combo: QComboBox, value: object) -> bool:
            return isinstance(value, str) and any(
                value == combo.itemText(i) for i in range(combo.count())
            )

        plot_type = cfg.get("plot_type")
        if _has_value(self.plot_type_combo, plot_type):
            self.plot_type_combo.setCurrentText(str(plot_type))
        self._toggle_error_mode_enabled(self.plot_type_combo.currentText())

        y_axis = cfg.get("y_axis")
        if _has_value(self.y_combo, y_axis):
            self.y_combo.setCurrentText(str(y_axis))

        x_axis = cfg.get("x_axis")
        if _has_value(self.x_combo, x_axis):
            self.x_combo.setCurrentText(str(x_axis))

        err_mode = cfg.get("error_mode")
        if _has_value(self.error_mode_combo, err_mode):
            self.error_mode_combo.setCurrentText(str(err_mode))

        x_limits = cfg.get("x_limits") or (None, None)
        if isinstance(x_limits, (list, tuple)) and len(x_limits) == 2:
            self._set_limit_text(self.xmin_edit, x_limits[0])
            self._set_limit_text(self.xmax_edit, x_limits[1])

        y_limits = cfg.get("y_limits") or (None, None)
        if isinstance(y_limits, (list, tuple)) and len(y_limits) == 2:
            self._set_limit_text(self.ymin_edit, y_limits[0])
            self._set_limit_text(self.ymax_edit, y_limits[1])

    def _parse_limit(self, edit: QLineEdit) -> Optional[float]:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _set_limit_text(self, edit: QLineEdit, value: object) -> None:
        if value is None or value == "":
            edit.clear()
        else:
            edit.setText(str(value))

    def _emit_plot(self) -> None:
        if not self.y_combo.currentText():
            return
        config = {
            "x_axis": self.x_combo.currentText(),
            "y_axis": self.y_combo.currentText(),
            "plot_type": self.plot_type_combo.currentText(),
            "error_mode": self.error_mode_combo.currentText(),
            "x_limits": (self._parse_limit(self.xmin_edit), self._parse_limit(self.xmax_edit)),
            "y_limits": (self._parse_limit(self.ymin_edit), self._parse_limit(self.ymax_edit)),
        }
        self.plotRequested.emit(config)

    def _toggle_error_mode_enabled(self, plot_type: str) -> None:
        self.error_mode_combo.setEnabled(plot_type == "Scatter")
