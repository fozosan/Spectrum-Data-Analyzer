"""Trendline configuration form."""
from __future__ import annotations

from typing import List, Optional, Tuple

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
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
    exportResidualsRequested = pyqtSignal()
    exportFitsRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.fit_label = QLabel("No fit computed", self)

        # Data trendline controls
        self.data_model_combo = QComboBox(self)
        self.data_model_combo.addItems(["Linear", "Quadratic", "Power"])
        self.fit_button = QPushButton("Fit Data", self)

        fit_group = QGroupBox("Data Trendline", self)
        fit_layout = QVBoxLayout(fit_group)
        fit_layout.addWidget(self.data_model_combo)
        fit_layout.addWidget(self.fit_button)
        fit_layout.addWidget(self.fit_label)

        # Literature trendline controls
        self.lit_model_combo = QComboBox(self)
        self.lit_model_combo.addItems(["Linear", "Quadratic", "Power"])
        self.lit_coeff1 = QLineEdit(self)
        self.lit_coeff2 = QLineEdit(self)
        self.lit_coeff3 = QLineEdit(self)
        self.lit_overlay_button = QPushButton("Overlay Literature", self)

        literature_form = QFormLayout()
        literature_form.addRow("Model", self.lit_model_combo)
        coeff_row = QHBoxLayout()
        coeff_row.addWidget(self.lit_coeff1)
        coeff_row.addWidget(self.lit_coeff2)
        coeff_row.addWidget(self.lit_coeff3)
        literature_form.addRow("Coefficients", coeff_row)
        literature_form.addRow(self.lit_overlay_button)

        literature_group = QGroupBox("Literature Trendline", self)
        literature_group.setLayout(literature_form)

        # Intersections table and controls
        self.intersections_table = QTableWidget(self)
        self.intersections_table.setColumnCount(2)
        self.intersections_table.setHorizontalHeaderLabels(["x", "y"])
        self.intersections_table.verticalHeader().setVisible(False)
        self.intersections_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.intersections_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.intersections_table.setSelectionMode(QTableWidget.SingleSelection)

        self.intersections_btn = QPushButton("Compute Intersections", self)
        self.export_intersections_btn = QPushButton("Export Intersections CSV", self)

        inter_btn_row = QHBoxLayout()
        inter_btn_row.addWidget(self.intersections_btn)
        inter_btn_row.addWidget(self.export_intersections_btn)

        self.export_residuals_btn = QPushButton("Export Residuals CSV", self)
        self.export_fits_btn = QPushButton("Export Fit Parameters CSV", self)

        export_row = QHBoxLayout()
        export_row.addWidget(self.export_residuals_btn)
        export_row.addWidget(self.export_fits_btn)

        # Compose layout
        root = QVBoxLayout(self)
        root.addWidget(fit_group)
        root.addWidget(literature_group)
        root.addLayout(inter_btn_row)
        root.addWidget(self.intersections_table)
        root.addLayout(export_row)
        root.addStretch(1)

        # Wire signals
        self.fit_button.clicked.connect(self._emit_fit_requested)
        self.lit_model_combo.currentTextChanged.connect(self._on_lit_model_changed)
        self.lit_overlay_button.clicked.connect(self._emit_literature_overlay)
        self.intersections_btn.clicked.connect(self.intersectionsRequested.emit)
        self.export_intersections_btn.clicked.connect(
            self.exportIntersectionsRequested.emit
        )
        self.export_residuals_btn.clicked.connect(self.exportResidualsRequested.emit)
        self.export_fits_btn.clicked.connect(self.exportFitsRequested.emit)

        # Initialize default state
        self._on_lit_model_changed(self.lit_model_combo.currentText())

    # --------------------------- Public API ---------------------------
    def set_fit_summary(self, text: str) -> None:
        """Update the fit summary label."""

        self.fit_label.setText(text or "No fit computed")

    def set_intersections(self, points: List[Tuple[float, float]]) -> None:
        """Populate the intersections table with the provided points."""

        pts = points or []
        self.intersections_table.setRowCount(len(pts))
        for row, (x, y) in enumerate(pts):
            self.intersections_table.setItem(row, 0, QTableWidgetItem(f"{x:.6g}"))
            self.intersections_table.setItem(row, 1, QTableWidgetItem(f"{y:.6g}"))
        self.intersections_table.resizeColumnsToContents()

    # --------------------------- Internal helpers ---------------------------
    def _emit_fit_requested(self) -> None:
        model = (self.data_model_combo.currentText() or "Linear").strip()
        self.fitRequested.emit(model)

    def _on_lit_model_changed(self, model: str) -> None:
        self._update_lit_placeholders(model)
        m = (model or "").lower()
        if m == "quadratic":
            self.lit_coeff3.setVisible(True)
            self.lit_coeff3.setEnabled(True)
        else:
            self.lit_coeff3.setVisible(False)
            self.lit_coeff3.setEnabled(False)
            self.lit_coeff3.clear()

    def _emit_literature_overlay(self) -> None:
        model = (self.lit_model_combo.currentText() or "").strip().lower()
        try:
            if model == "linear":
                m = float((self.lit_coeff1.text() or "").strip())
                b = float((self.lit_coeff2.text() or "").strip())
                payload = {"model": "linear", "coeffs": (m, b)}
            elif model == "quadratic":
                a = float((self.lit_coeff1.text() or "").strip())
                b = float((self.lit_coeff2.text() or "").strip())
                c = float((self.lit_coeff3.text() or "").strip())
                payload = {"model": "quadratic", "coeffs": (a, b, c)}
            elif model == "power":
                A = float((self.lit_coeff1.text() or "").strip())
                B = float((self.lit_coeff2.text() or "").strip())
                payload = {"model": "power", "coeffs": (A, B)}
            else:
                QMessageBox.warning(self, "Literature", "Unsupported model.")
                return
        except ValueError:
            QMessageBox.warning(
                self,
                "Literature",
                "Please enter valid numeric coefficients for the selected model.",
            )
            return
        self.literatureOverlayRequested.emit(payload)

    def _update_lit_placeholders(self, model: str) -> None:
        m = (model or "").lower()
        if m == "linear":
            self.lit_coeff1.setPlaceholderText("Slope m")
            self.lit_coeff2.setPlaceholderText("Intercept b")
            self.lit_coeff3.setPlaceholderText("")
        elif m == "quadratic":
            self.lit_coeff1.setPlaceholderText("a (x^2)")
            self.lit_coeff2.setPlaceholderText("b (x)")
            self.lit_coeff3.setPlaceholderText("c")
        else:
            self.lit_coeff1.setPlaceholderText("A (scale)")
            self.lit_coeff2.setPlaceholderText("B (exponent)")
            self.lit_coeff3.setPlaceholderText("")
