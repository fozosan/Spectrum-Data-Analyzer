"""Widget to configure and compute Raman metrics."""
from __future__ import annotations

from typing import List

import pandas as pd
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

from raman_analyzer.analysis.metrics import (
    assemble_results,
    compute_difference,
    compute_normalized_area,
    compute_ratio,
    compute_single,
)
from raman_analyzer.models.selections import PeakSelector


class SelectorEditor(QGroupBox):
    """Allow the user to configure a :class:`PeakSelector`."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["by_index", "nearest_center"])
        self.values_edit = QLineEdit(self)
        self.values_edit.setPlaceholderText("Comma-separated values")
        self.tolerance_edit = QLineEdit(self)
        self.tolerance_edit.setPlaceholderText("Tolerance (cm^-1)")
        self.tolerance_edit.setText("10")

        form = QFormLayout(self)
        form.addRow("Mode", self.mode_combo)
        form.addRow("Values", self.values_edit)
        form.addRow("Tolerance", self.tolerance_edit)

        self.mode_combo.currentTextChanged.connect(self._update_tolerance_state)
        self._update_tolerance_state(self.mode_combo.currentText())

    def _update_tolerance_state(self, mode: str) -> None:
        self.tolerance_edit.setEnabled(mode == "nearest_center")

    def _parse_values(self) -> List[float]:
        text = self.values_edit.text().strip()
        if not text:
            return []
        parts = [part.strip() for part in text.split(",") if part.strip()]
        values = []
        for part in parts:
            try:
                values.append(float(part))
            except ValueError:
                continue
        return values

    def to_selector(self) -> PeakSelector:
        mode = self.mode_combo.currentText()
        values = self._parse_values()
        if mode == "by_index":
            indices = [int(v) for v in values]
            return PeakSelector(mode="by_index", indices=indices)
        tolerance = float(self.tolerance_edit.text() or 10.0)
        return PeakSelector(mode="nearest_center", centers=values, tolerance_cm1=tolerance)


class CalcBuilderWidget(QWidget):
    """Allow the user to compute metrics and emit the results."""

    metricComputed = pyqtSignal(str, pd.DataFrame)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.available_attributes: List[str] = [
            "area",
            "height",
            "fwhm",
            "center",
            "area_pct",
        ]
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Single", "Ratio", "Difference", "Normalized Area"])
        self.attr_combo = QComboBox(self)
        self.attr_combo.addItems(self.available_attributes)
        self.agg_combo = QComboBox(self)
        self.agg_combo.addItems(["sum", "mean"])
        self.metric_name_edit = QLineEdit(self)
        self.metric_name_edit.setPlaceholderText("Metric name (auto if blank)")

        self.norm_combo = QComboBox(self)
        self.norm_combo.addItems(["To total area", "To Selection B"])
        self.norm_combo.setToolTip(
            "Normalize selected area to total area in the file, or to Selection B."
        )

        self.selector_a = SelectorEditor("Selection A", self)
        self.selector_b = SelectorEditor("Selection B", self)

        self.compute_button = QPushButton("Compute", self)

        form = QFormLayout()
        form.addRow("Mode", self.mode_combo)
        form.addRow("Attribute", self.attr_combo)
        form.addRow("Aggregation", self.agg_combo)
        form.addRow("Metric Name", self.metric_name_edit)
        form.addRow("Normalization", self.norm_combo)
        self._norm_label = form.labelForField(self.norm_combo)

        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self.selector_a)
        selector_layout.addWidget(self.selector_b)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(selector_layout)
        layout.addWidget(self.compute_button)
        layout.addStretch(1)

        self.compute_button.clicked.connect(self._compute)
        self.mode_combo.currentTextChanged.connect(self._toggle_norm_visibility)
        self._toggle_norm_visibility(self.mode_combo.currentText())

        self.raw_df: pd.DataFrame | None = None
        self.file_to_tag: dict[str, str] = {}

    def set_available_attributes(self, attrs: List[str]) -> None:
        self.available_attributes = attrs
        current = self.attr_combo.currentText()
        self.attr_combo.clear()
        self.attr_combo.addItems(attrs)
        if current in attrs:
            self.attr_combo.setCurrentText(current)

    def set_data(self, df: pd.DataFrame, file_to_tag: dict[str, str]) -> None:
        self.raw_df = df
        self.file_to_tag = file_to_tag

    def _auto_metric_name(self, base_attr: str) -> str:
        mode = self.mode_combo.currentText().lower()
        if mode == "normalized area":
            suffix = "total" if self.norm_combo.currentText().startswith("To total") else "selB"
            return f"norm_area_{suffix}"
        return f"{mode}_{base_attr}".replace(" ", "_")

    def _compute(self) -> None:
        if self.raw_df is None or self.raw_df.empty:
            return
        attr = self.attr_combo.currentText()
        mode = self.mode_combo.currentText()
        metric_name = self.metric_name_edit.text().strip() or self._auto_metric_name(attr)
        agg = self.agg_combo.currentText()

        selector_a = self.selector_a.to_selector()
        selector_b = self.selector_b.to_selector()

        if mode == "Single":
            result = compute_single(self.raw_df, attr, selector_a, agg=agg)
        elif mode == "Ratio":
            result = compute_ratio(self.raw_df, attr, selector_a, selector_b, agg=agg)
        elif mode == "Difference":
            result = compute_difference(self.raw_df, attr, selector_a, selector_b)
        else:
            if self.norm_combo.currentText().startswith("To total"):
                result = compute_normalized_area(
                    self.raw_df,
                    selector_a,
                    reference="total",
                    agg=agg,
                )
            else:
                result = compute_normalized_area(
                    self.raw_df,
                    selector_a,
                    reference="selection",
                    ref_selector=selector_b,
                    agg=agg,
                )

        long_df = assemble_results(result, self.file_to_tag, metric_name)
        self.metricComputed.emit(metric_name, long_df)

    def _toggle_norm_visibility(self, mode: str) -> None:
        visible = mode == "Normalized Area"
        self.norm_combo.setVisible(visible)
        if self._norm_label is not None:
            self._norm_label.setVisible(visible)
        self.attr_combo.setEnabled(not visible)
        if visible:
            for i in range(self.attr_combo.count()):
                if self.attr_combo.itemText(i) == "area":
                    self.attr_combo.setCurrentIndex(i)
                    break
