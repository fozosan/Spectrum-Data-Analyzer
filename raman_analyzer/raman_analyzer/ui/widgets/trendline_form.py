"""Trendline configuration form."""
from __future__ import annotations

from typing import List, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class TrendlineForm(QWidget):
    """Collect configuration for data and literature trendlines."""

    fitRequested = pyqtSignal(str)
    literatureOverlayRequested = pyqtSignal(dict)
    intersectionsRequested = pyqtSignal()
    exportIntersectionsRequested = pyqtSignal()
    exportFitsRequested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.fit_label = QLabel("No fit computed", self)

        # Data fit controls
        self.data_model_combo = QComboBox(self)
        self.data_model_combo.addItems(["Linear", "Quadratic", "Power"])
        self.fit_button = QPushButton("Fit Data", self)

        fit_group = QGroupBox("Data Trendline", self)
        fit_layout = QVBoxLayout(fit_group)
        fit_layout.addWidget(self.data_model_combo)
        fit_layout.addWidget(self.fit_button)
        fit_layout.addWidget(self.fit_label)

        # Literature controls
        self.lit_model_combo = QComboBox(self)
        self.lit_model_combo.addItems(["Linear", "Quadratic", "Power"])
        self.lit_coeff1 = QLineEdit(self)
        self.lit_coeff2 = QLineEdit(self)
        self.lit_coeff3 = QLineEdit(self)
        self._update_lit_placeholders("Linear")
        self.lit_overlay_button = QPushButton("Overlay Literature", self)

        literature_form = QFormLayout()
        literature_form.addRow("Model", self.lit_model_combo)
        literature_form.addRow("Coeff 1", self.lit_coeff1)
        literature_form.addRow("Coeff 2", self.lit_coeff2)
        literature_form.addRow("Coeff 3", self.lit_coeff3)
        literature_form.addRow(self.lit_overlay_button)

        literature_group = QGroupBox("Literature Trendline", self)
        literature_group.setLayout(literature_form)

        # Intersections and export
        self.intersections_button = QPushButton("Compute Intersections", self)
        self.export_intersections_button = QPushButton("Export Intersections CSV", self)
        self.export_fits_button = QPushButton("Export Fit Params CSV", self)

        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["X", "Y"])
        self.table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addWidget(fit_group)
        layout.addWidget(literature_group)
        layout.addWidget(self.intersections_button)
        layout.addWidget(self.export_intersections_button)
        layout.addWidget(self.export_fits_button)
        layout.addWidget(self.table)

        self.fit_button.clicked.connect(self._emit_fit)
        self.lit_overlay_button.clicked.connect(self._emit_literature)
        self.lit_model_combo.currentTextChanged.connect(self._update_lit_placeholders)
        self.intersections_button.clicked.connect(self.intersectionsRequested.emit)
        self.export_intersections_button.clicked.connect(
            self.exportIntersectionsRequested.emit
        )
        self.export_fits_button.clicked.connect(self.exportFitsRequested.emit)

    def _emit_fit(self) -> None:
        model = self.data_model_combo.currentText().lower()
        self.fitRequested.emit(model)

    def _emit_literature(self) -> None:
        model = self.lit_model_combo.currentText().lower()
        try:
            if model == "linear":
                coeffs = (float(self.lit_coeff1.text()), float(self.lit_coeff2.text()))
            elif model == "quadratic":
                coeffs = (
                    float(self.lit_coeff1.text()),
                    float(self.lit_coeff2.text()),
                    float(self.lit_coeff3.text()),
                )
            else:
                coeffs = (
                    float(self.lit_coeff1.text()),
                    float(self.lit_coeff2.text()),
                )
        except ValueError:
            return
        payload = {"model": model, "coeffs": coeffs}
        self.literatureOverlayRequested.emit(payload)

    def set_fit_summary(self, summary: str) -> None:
        self.fit_label.setText(summary)

    def set_intersections(self, points: List[Tuple[float, float]]) -> None:
        self.table.setRowCount(len(points))
        for row, (x, y) in enumerate(points):
            self.table.setItem(row, 0, QTableWidgetItem(f"{x:.4f}"))
            self.table.setItem(row, 1, QTableWidgetItem(f"{y:.4f}"))
        if not points:
            self.table.setRowCount(0)

    # --- helpers
    def _update_lit_placeholders(self, model: str) -> None:
        model = model.lower()
        if model == "linear":
            self.lit_coeff1.setPlaceholderText("Slope m")
            self.lit_coeff2.setPlaceholderText("Intercept b")
            self.lit_coeff3.setPlaceholderText("unused")
            self.lit_coeff3.setEnabled(False)
        elif model == "quadratic":
            self.lit_coeff1.setPlaceholderText("a")
            self.lit_coeff2.setPlaceholderText("b")
            self.lit_coeff3.setPlaceholderText("c")
            self.lit_coeff3.setEnabled(True)
        else:
            self.lit_coeff1.setPlaceholderText("A  (y = A*x^B)")
            self.lit_coeff2.setPlaceholderText("B")
            self.lit_coeff3.setPlaceholderText("unused")
            self.lit_coeff3.setEnabled(False)
