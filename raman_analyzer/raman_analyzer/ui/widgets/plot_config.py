"""Plot configuration controls."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.metrics: List[str] = []
        self.x_combo = QComboBox(self)
        self.y_combo = QComboBox(self)
        self.plot_type_combo = QComboBox(self)
        self.plot_type_combo.addItems(["Scatter", "Box"])
        self.error_bars_check = QCheckBox("Show error bars", self)

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
        form.addRow("Error bars", self.error_bars_check)

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
        button_layout.addWidget(self.plot_button)
        button_layout.addWidget(self.export_plot_button)
        button_layout.addWidget(self.export_metrics_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(axis_group)
        layout.addLayout(button_layout)
        layout.addStretch(1)

        self.plot_button.clicked.connect(self._emit_plot)
        self.export_plot_button.clicked.connect(self.exportPlotRequested.emit)
        self.export_metrics_button.clicked.connect(self.exportMetricsRequested.emit)

    def set_metrics(self, metrics: List[str]) -> None:
        """Update the metric dropdowns."""

        self.metrics = metrics
        current_y = self.y_combo.currentText()
        current_x = self.x_combo.currentText()

        self.y_combo.clear()
        self.y_combo.addItems(metrics)

        self.x_combo.clear()
        self.x_combo.addItem("Group (categorical)")
        self.x_combo.addItems(metrics)

        if current_y in metrics:
            self.y_combo.setCurrentText(current_y)
        if current_x and (current_x == "Group (categorical)" or current_x in metrics):
            self.x_combo.setCurrentText(current_x)

    def _parse_limit(self, edit: QLineEdit) -> Optional[float]:
        text = edit.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _emit_plot(self) -> None:
        if not self.y_combo.currentText():
            return
        config = {
            "x_axis": self.x_combo.currentText(),
            "y_axis": self.y_combo.currentText(),
            "plot_type": self.plot_type_combo.currentText(),
            "error_bars": self.error_bars_check.isChecked(),
            "x_limits": (self._parse_limit(self.xmin_edit), self._parse_limit(self.xmax_edit)),
            "y_limits": (self._parse_limit(self.ymin_edit), self._parse_limit(self.ymax_edit)),
        }
        self.plotRequested.emit(config)
