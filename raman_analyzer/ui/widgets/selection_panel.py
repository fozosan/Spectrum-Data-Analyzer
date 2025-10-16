"""Manual Selection Panel: build A/B selections from raw CSV cells.

Modes:
  - Single:     A, B
  - Ratio:      A(num/den), B(num/den)
  - Difference: A(left/right), B(left/right)

Features:
  - Arm a target, then double-click raw grid to add a pick (row/col are 1-based).
  - Auto-populate a target across Selected/All files by row/col.
  - Allow multiple picks per component; aggregate via Mean or Sum.
  - Immediately compute & emit per-file values for A and B after any change.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import math
import statistics

import pandas as pd
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class SelectionPanel(QWidget):
    """UI for building Selection A/B from manual picks and auto-populate."""

    # Emitted whenever computed metrics change. Each payload is a tuple (metric_name: str, df: pd.DataFrame)
    # where df has columns ['file', 'value'].
    metricsUpdated = pyqtSignal(object, object)
    # Ask MainWindow to auto-populate a target ("A.single" / "A.num" / "A.den" / ...)
    autopopulateRequested = pyqtSignal(str, int, int, str)  # target, row1, col1, scope: 'Selected'|'All'

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._mode = "Single"  # 'Single' | 'Ratio' | 'Difference'
        self._armed: str = "A.single"  # one of target keys
        self._file_to_tag: Dict[str, str] = {}
        # picks structure: { 'A': {'single': {file: [(r,c,val), ...]} , 'num': {...}, 'den': {...}, 'left': {...}, 'right': {...}},
        #                    'B': {...} }
        self._picks: Dict[str, Dict[str, Dict[str, List[Tuple[int, int, float]]]]] = (
            self._blank_picks()
        )

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["Single", "Ratio", "Difference"])

        self.agg_combo = QComboBox(self)
        self.agg_combo.addItems(["Mean", "Sum"])
        self.agg_combo.setToolTip("How to combine multiple picks per component.")

        # Target radios (rebuilt on mode change)
        self.targets_box = QGroupBox("Arm target — double-click grid to add", self)
        self.targets_layout = QVBoxLayout(self.targets_box)
        self._target_radios: List[QRadioButton] = []
        self._target_radio_map: Dict[str, QRadioButton] = {}
        self.armed_label = QLabel("", self)

        # Auto-populate controls
        self.row_spin = QSpinBox(self)
        self.row_spin.setMinimum(1)
        self.col_spin = QSpinBox(self)
        self.col_spin.setMinimum(1)
        self.scope_combo = QComboBox(self)
        self.scope_combo.addItems(["Selected", "All"])
        self.autofill_btn = QPushButton("Auto-populate into target", self)

        # A/B tables
        self.tableA = QTableWidget(self)
        self.tableB = QTableWidget(self)
        for table in (self.tableA, self.tableB):
            table.setColumnCount(7)
            table.setHorizontalHeaderLabels(["file", "tag", "component", "row", "col", "value", "count/agg"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.SingleSelection)

        # Computed value previews (per file)
        self.previewA = QTableWidget(self)
        self.previewB = QTableWidget(self)
        for table in (self.previewA, self.previewB):
            table.setColumnCount(3)
            table.setHorizontalHeaderLabels(["file", "tag", "value"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.NoSelection)

        self.removeA_btn = QPushButton("Remove Selected (A)", self)
        self.clearA_btn = QPushButton("Clear A", self)
        self.removeB_btn = QPushButton("Remove Selected (B)", self)
        self.clearB_btn = QPushButton("Clear B", self)

        # Layout
        config_box = QGroupBox("Manual Selection (Single / Ratio / Difference)", self)
        config_form = QFormLayout(config_box)
        config_form.addRow("Mode", self.mode_combo)
        config_form.addRow("Aggregator", self.agg_combo)

        autopop_row = QHBoxLayout()
        autopop_row.addWidget(QLabel("Row:"))
        autopop_row.addWidget(self.row_spin)
        autopop_row.addWidget(QLabel("Col:"))
        autopop_row.addWidget(self.col_spin)
        autopop_row.addWidget(QLabel("Across:"))
        autopop_row.addWidget(self.scope_combo)
        autopop_row.addWidget(self.autofill_btn)

        lists_row = QHBoxLayout()
        colA = QVBoxLayout()
        colA.addWidget(QLabel("Selection A"))
        colA.addWidget(self.tableA)
        rowA = QHBoxLayout()
        rowA.addWidget(self.removeA_btn)
        rowA.addWidget(self.clearA_btn)
        colA.addLayout(rowA)
        colA.addWidget(QLabel("Computed A (per file)"))
        colA.addWidget(self.previewA)
        colB = QVBoxLayout()
        colB.addWidget(QLabel("Selection B"))
        colB.addWidget(self.tableB)
        rowB = QHBoxLayout()
        rowB.addWidget(self.removeB_btn)
        rowB.addWidget(self.clearB_btn)
        colB.addLayout(rowB)
        colB.addWidget(QLabel("Computed B (per file)"))
        colB.addWidget(self.previewB)
        lists_row.addLayout(colA)
        lists_row.addLayout(colB)

        root = QVBoxLayout(self)
        root.addWidget(config_box)
        root.addWidget(self.targets_box)
        root.addWidget(self.armed_label)
        root.addLayout(autopop_row)
        root.addLayout(lists_row)
        root.addStretch(1)

        # Wire
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.agg_combo.currentTextChanged.connect(lambda _: self._recompute_and_emit())
        self.autofill_btn.clicked.connect(self._on_autofill)
        self.removeA_btn.clicked.connect(lambda: self._remove_selected("A"))
        self.clearA_btn.clicked.connect(lambda: self._clear_bucket("A"))
        self.removeB_btn.clicked.connect(lambda: self._remove_selected("B"))
        self.clearB_btn.clicked.connect(lambda: self._clear_bucket("B"))

        # Initialize mode
        self._rebuild_target_radios("Single")
        self._update_armed_ui()

    # ----------------------------- Public API -----------------------------
    def set_context(self, file_to_tag: Dict[str, str]) -> None:
        self._file_to_tag = dict(file_to_tag or {})

    def get_state(self) -> dict:
        """Return a JSON-serializable snapshot of the current selections."""

        def _serialize_bucket(bucket: str) -> Dict[str, Dict[str, List[List[float]]]]:
            serialized: Dict[str, Dict[str, List[List[float]]]] = {}
            for comp in ("single", "num", "den", "left", "right"):
                comp_map: Dict[str, List[List[float]]] = {}
                for file_id, picks in self._picks[bucket][comp].items():
                    comp_map[file_id] = [
                        [int(r), int(c), float(v)]
                        for r, c, v in picks
                    ]
                serialized[comp] = comp_map
            return serialized

        return {
            "mode": self._mode,
            "aggregator": self.agg_combo.currentText(),
            "picks": {
                "A": _serialize_bucket("A"),
                "B": _serialize_bucket("B"),
            },
            "armed": self._armed,
        }

    def apply_state(self, state: dict) -> None:
        """Restore selections and configuration from a saved snapshot."""

        if not isinstance(state, dict):
            return

        requested_mode = str(state.get("mode") or "Single")
        if requested_mode not in {"Single", "Ratio", "Difference"}:
            requested_mode = "Single"
        if requested_mode != self._mode:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(requested_mode)
            self.mode_combo.blockSignals(False)
        self._mode = requested_mode
        self._rebuild_target_radios(requested_mode)
        self._picks = self._blank_picks()

        aggregator = state.get("aggregator")
        self.agg_combo.blockSignals(True)
        try:
            if isinstance(aggregator, str) and aggregator in {"Mean", "Sum"}:
                self.agg_combo.setCurrentText(aggregator)
            else:
                self.agg_combo.setCurrentText("Mean")
        finally:
            self.agg_combo.blockSignals(False)

        # Prepare restored picks structure
        restored = self._blank_picks()

        picks = state.get("picks")
        if isinstance(picks, dict):
            for bucket in ("A", "B"):
                bucket_src = picks.get(bucket)
                if not isinstance(bucket_src, dict):
                    continue
                for comp in ("single", "num", "den", "left", "right"):
                    comp_map = bucket_src.get(comp)
                    if not isinstance(comp_map, dict):
                        continue
                    clean: Dict[str, List[Tuple[int, int, float]]] = {}
                    for file_id, entries in comp_map.items():
                        if not isinstance(entries, list):
                            continue
                        cleaned: List[Tuple[int, int, float]] = []
                        for entry in entries:
                            if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                                continue
                            try:
                                r_val = int(entry[0])
                                c_val = int(entry[1])
                                v_val = float(entry[2])
                            except (TypeError, ValueError):
                                continue
                            cleaned.append((r_val, c_val, v_val))
                        if cleaned:
                            clean[str(file_id)] = cleaned
                    restored[bucket][comp] = clean

        self._picks = restored

        armed = state.get("armed")
        if isinstance(armed, str):
            valid_targets = {
                "Single": {"A.single", "B.single"},
                "Ratio": {"A.num", "A.den", "B.num", "B.den"},
                "Difference": {"A.left", "A.right", "B.left", "B.right"},
            }.get(self._mode, set())
            if armed in valid_targets:
                self._set_armed_radio(armed)

        self._refresh_tables()
        self._recompute_and_emit()

    def add_pick(
        self,
        file_id: str,
        row1: int,
        col1: int,
        value: float,
        *,
        target: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Add a single manual pick to the desired target (default: armed)."""

        key = target or self._armed  # e.g., "A.single", "A.num", ...
        bucket, comp = key.split(".")
        picks = self._picks[bucket][comp].setdefault(file_id, [])
        picks.append((int(row1), int(col1), float(value)))
        if tag is not None:
            self._file_to_tag[str(file_id)] = str(tag)
        self._refresh_tables()
        self._recompute_and_emit()

    # ----------------------------- Internal UI -----------------------------
    def _on_mode_changed(self, new_mode: str) -> None:
        self._mode = new_mode
        self._rebuild_target_radios(new_mode)
        # Clear all picks when mode changes (keeps semantics unambiguous).
        self._picks = self._blank_picks()
        self._refresh_tables()
        self._recompute_and_emit()

    def _rebuild_target_radios(self, mode: str) -> None:
        # Clear layout
        for rb in self._target_radios:
            rb.deleteLater()
        self._target_radios = []
        self._target_radio_map = {}
        while self.targets_layout.count():
            item = self.targets_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        def _add_radio(label: str, key: str, checked: bool = False) -> None:
            rb = QRadioButton(label, self.targets_box)
            rb.setChecked(checked)
            rb.toggled.connect(lambda state, k=key: self._on_arm(k, state))
            self._target_radios.append(rb)
            self.targets_layout.addWidget(rb)
            self._target_radio_map[key] = rb

        if mode == "Single":
            _add_radio("A (single)", "A.single", True)
            _add_radio("B (single)", "B.single", False)
            self._armed = "A.single"
        elif mode == "Ratio":
            _add_radio("A: Numerator", "A.num", True)
            _add_radio("A: Denominator", "A.den", False)
            _add_radio("B: Numerator", "B.num", False)
            _add_radio("B: Denominator", "B.den", False)
            self._armed = "A.num"
        else:  # Difference
            _add_radio("A: Left", "A.left", True)
            _add_radio("A: Right", "A.right", False)
            _add_radio("B: Left", "B.left", False)
            _add_radio("B: Right", "B.right", False)
            self._armed = "A.left"
        self._update_armed_ui()

    def _on_arm(self, key: str, checked: bool) -> None:
        if checked:
            self._armed = key
            self._update_armed_ui()

    def _set_armed_radio(self, key: str) -> None:
        self._armed = key
        radio = self._target_radio_map.get(key)
        if radio is not None and not radio.isChecked():
            radio.setChecked(True)
        self._update_armed_ui()

    def _on_autofill(self) -> None:
        target = self._armed
        row1 = int(self.row_spin.value())
        col1 = int(self.col_spin.value())
        scope = self.scope_combo.currentText()  # 'Selected' or 'All'
        self.autopopulateRequested.emit(target, row1, col1, scope)
        # Button text reflects the armed target; no further updates needed here.

    def _update_armed_ui(self) -> None:
        """Update helper text and button label for the currently armed target."""
        friendly_labels = {
            "A.single": "A (single)",
            "B.single": "B (single)",
            "A.num": "A: Numerator",
            "A.den": "A: Denominator",
            "B.num": "B: Numerator",
            "B.den": "B: Denominator",
            "A.left": "A: Left",
            "A.right": "A: Right",
            "B.left": "B: Left",
            "B.right": "B: Right",
        }
        label = friendly_labels.get(self._armed, self._armed.replace(".", ": "))
        self.armed_label.setText(f"Target: {label}")
        self.autofill_btn.setText(f"Auto-populate into {label}")

    def _blank_picks(self) -> Dict[str, Dict[str, Dict[str, List[Tuple[int, int, float]]]]]:
        components = ("single", "num", "den", "left", "right")
        return {
            bucket: {comp: {} for comp in components}
            for bucket in ("A", "B")
        }

    def _refresh_tables(self) -> None:
        self._fill_table(self.tableA, "A")
        self._fill_table(self.tableB, "B")

    def _fill_table(self, table: QTableWidget, bucket: str) -> None:
        rows: List[Tuple[str, str, str, int, int, float]] = []
        for comp in ("single", "num", "den", "left", "right"):
            comp_map = self._picks[bucket][comp]
            for file_id, items in comp_map.items():
                tag = self._file_to_tag.get(file_id, "")
                for r1, c1, val in items:
                    rows.append((file_id, tag, comp, r1, c1, val))
        table.setRowCount(len(rows))
        for i, (file_id, tag, comp, r1, c1, val) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(str(file_id)))
            table.setItem(i, 1, QTableWidgetItem(str(tag)))
            table.setItem(i, 2, QTableWidgetItem(comp))
            table.setItem(i, 3, QTableWidgetItem(str(r1)))
            table.setItem(i, 4, QTableWidgetItem(str(c1)))
            table.setItem(i, 5, QTableWidgetItem(f"{val:.6g}"))
            table.setItem(i, 6, QTableWidgetItem(""))
        table.resizeColumnsToContents()

    def _clear_bucket(self, bucket: str) -> None:
        self._picks[bucket] = {"single": {}, "num": {}, "den": {}, "left": {}, "right": {}}
        self._refresh_tables()
        self._recompute_and_emit()

    def _remove_selected(self, bucket: str) -> None:
        """Remove the highlighted row from Selection A/B (exact pick deletion)."""
        table = self.tableA if bucket == "A" else self.tableB
        row = table.currentRow()
        if row < 0:
            return
        file_item = table.item(row, 0)
        comp_item = table.item(row, 2)
        r_item = table.item(row, 3)
        c_item = table.item(row, 4)
        v_item = table.item(row, 5)
        if not (file_item and comp_item and r_item and c_item and v_item):
            return
        file_id = file_item.text()
        comp = comp_item.text()
        try:
            r1 = int(r_item.text())
            c1 = int(c_item.text())
            val = float(v_item.text())
        except Exception:
            return
        lst = self._picks[bucket].get(comp, {}).get(file_id, [])
        for idx, (rr, cc, vv) in enumerate(lst):
            if rr == r1 and cc == c1 and abs(vv - val) <= 1e-12:
                del lst[idx]
                break
        if not lst and file_id in self._picks[bucket].get(comp, {}):
            self._picks[bucket][comp].pop(file_id, None)
        self._refresh_tables()
        self._recompute_and_emit()

    # ----------------------------- Compute -----------------------------
    def _aggregate(self, values: List[float]) -> Optional[float]:
        if not values:
            return None
        mode = self.agg_combo.currentText()
        if mode == "Sum":
            return float(sum(values))
        try:
            return float(statistics.fmean(values))
        except Exception:
            return float(sum(values) / len(values))

    def _compute_bucket(self, bucket: str) -> pd.DataFrame:
        picks = self._picks[bucket]
        files = set()
        for comp in picks.values():
            files.update(comp.keys())
        results: List[Tuple[str, float]] = []
        for file_id in sorted(files):
            if self._mode == "Single":
                vals = [val for _, _, val in picks["single"].get(file_id, [])]
                agg = self._aggregate(vals)
                results.append((file_id, math.nan if agg is None else float(agg)))
            elif self._mode == "Ratio":
                num_vals = [val for _, _, val in picks["num"].get(file_id, [])]
                den_vals = [val for _, _, val in picks["den"].get(file_id, [])]
                num = self._aggregate(num_vals)
                den = self._aggregate(den_vals)
                if num is None or den is None or den == 0:
                    results.append((file_id, math.nan))
                else:
                    results.append((file_id, float(num) / float(den)))
            else:  # Difference
                left_vals = [val for _, _, val in picks["left"].get(file_id, [])]
                right_vals = [val for _, _, val in picks["right"].get(file_id, [])]
                left = self._aggregate(left_vals)
                right = self._aggregate(right_vals)
                if left is None or right is None:
                    results.append((file_id, math.nan))
                else:
                    results.append((file_id, float(left) - float(right)))
        if not results:
            return pd.DataFrame(columns=["file", "value"])
        return pd.DataFrame(results, columns=["file", "value"])

    def _recompute_and_emit(self) -> None:
        a_df = self._compute_bucket("A")
        b_df = self._compute_bucket("B")
        a_name = "Selection A"
        b_name = "Selection B"
        self._update_count_agg_display(self.tableA, "A")
        self._update_count_agg_display(self.tableB, "B")
        self._refresh_preview(self.previewA, a_df)
        self._refresh_preview(self.previewB, b_df)
        self.metricsUpdated.emit((a_name, a_df), (b_name, b_df))

    def _update_count_agg_display(self, table: QTableWidget, bucket: str) -> None:
        look: Dict[Tuple[str, str], Tuple[int, Optional[float]]] = {}
        for comp in ("single", "num", "den", "left", "right"):
            comp_map = self._picks[bucket][comp]
            for file_id, items in comp_map.items():
                vals = [val for _, _, val in items]
                look[(file_id, comp)] = (len(vals), self._aggregate(vals))
        for row in range(table.rowCount()):
            file_item = table.item(row, 0)
            comp_item = table.item(row, 2)
            file_id = file_item.text() if file_item else ""
            comp = comp_item.text() if comp_item else ""
            count, agg = look.get((file_id, comp), (0, None))
            display = f"n={count}"
            if agg is not None:
                display += f" / {agg:.6g}"
            table.setItem(row, 6, QTableWidgetItem(display))

    def _refresh_preview(self, table: QTableWidget, df: pd.DataFrame) -> None:
        """Fill a compact 'file, tag, value' table for the computed bucket."""
        if df is None or df.empty:
            table.setRowCount(0)
            return
        rows: List[Tuple[str, str, object]] = []
        for _, series in df.iterrows():
            file_id = str(series.get("file", ""))
            value = series.get("value", math.nan)
            tag = self._file_to_tag.get(file_id, "")
            rows.append((file_id, tag, value))
        table.setRowCount(len(rows))
        for idx, (file_id, tag, value) in enumerate(rows):
            table.setItem(idx, 0, QTableWidgetItem(file_id))
            table.setItem(idx, 1, QTableWidgetItem(tag))
            if pd.isna(value):
                table.setItem(idx, 2, QTableWidgetItem(""))
            else:
                try:
                    formatted = f"{float(value):.6g}"
                except Exception:
                    formatted = str(value)
                table.setItem(idx, 2, QTableWidgetItem(formatted))
        table.resizeColumnsToContents()
