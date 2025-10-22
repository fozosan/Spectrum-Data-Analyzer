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
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QSplitter,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Minimum "visual rows" used to compute table minimum heights
DEFAULT_MIN_TABLE_ROWS = 8
# -----------------------------------------------------------------------------


class SelectionPanel(QWidget):
    """UI for building Selection A/B from manual picks and auto-populate."""

    # Emitted whenever computed metrics change. Each payload is a tuple (metric_name: str, df: pd.DataFrame)
    # where df has columns ['file', 'value'].
    metricsUpdated = pyqtSignal(object, object)
    # Ask MainWindow to auto-populate a target ("A.single" / "A.num" / "A.den" / ...)
    autopopulateRequested = pyqtSignal(str, int, int, str)  # target, row1, col1, scope: 'Selected'|'All'

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Claim space and grow when embedded inside nested splitters.
        self.setMinimumHeight(300)
        self.setMinimumWidth(520)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Tunables for default sizing
        self._MIN_CARD_HEIGHT = 240
        self._DEFAULT_LISTS_SIZES = [480, 260, 480, 260]

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
        self.mode_combo.setMinimumWidth(140)

        self.agg_combo = QComboBox(self)
        self.agg_combo.addItems(["Mean", "Sum"])
        self.agg_combo.setToolTip("How to combine multiple picks per component.")
        self.agg_combo.setMinimumWidth(140)

        # Target radios live in a dedicated box — we pre-create ALL radios once,
        # and later just show/hide the ones relevant to the current mode.
        self.targets_box = QGroupBox(
            "Armed target — double-click a value in the left grid to add", self
        )
        self.targets_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.targets_layout = QVBoxLayout(self.targets_box)
        self.targets_layout.setContentsMargins(8, 4, 8, 4)
        self.targets_layout.setSpacing(8)
        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._target_radio_map: Dict[str, QRadioButton] = {}
        self._target_radios: List[QRadioButton] = []

        def _mk_rb(key: str, label: str) -> QRadioButton:
            rb = QRadioButton(label, self.targets_box)
            self._button_group.addButton(rb)
            self.targets_layout.addWidget(rb)
            self._target_radio_map[key] = rb
            self._target_radios.append(rb)
            return rb

        _mk_rb("A.single", "A (single)")
        _mk_rb("B.single", "B (single)")
        _mk_rb("A.num", "A (numerator)")
        _mk_rb("A.den", "A (denominator)")
        _mk_rb("B.num", "B (numerator)")
        _mk_rb("B.den", "B (denominator)")
        _mk_rb("A.left", "A (left)")
        _mk_rb("A.right", "A (right)")
        _mk_rb("B.left", "B (left)")
        _mk_rb("B.right", "B (right)")
        self.targets_layout.addStretch(1)

        self._button_group.buttonToggled.connect(self._on_radio_toggled)
        self.armed_label = QLabel(
            "Double-click a value in the left grid to add to the armed target.", self
        )
        self.armed_label.setWordWrap(True)

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
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["file", "value"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.NoSelection)

        # Ensure all tables expand and scroll as needed inside splitters
        for table in (self.tableA, self.tableB, self.previewA, self.previewB):
            table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            table.setWordWrap(False)

        # Helper: enforce a minimum height of roughly five rows for tables
        def _enforce_min_rows(table: QTableWidget, rows: int = 5) -> None:
            vertical = table.verticalHeader()
            horizontal = table.horizontalHeader()
            row_height = vertical.defaultSectionSize()
            header_height = horizontal.height()
            table.setMinimumHeight(max(self._MIN_CARD_HEIGHT, header_height + row_height * rows + 20))
            horizontal.setSectionResizeMode(QHeaderView.Interactive)
            horizontal.setStretchLastSection(True)

        _enforce_min_rows(self.tableA, rows=DEFAULT_MIN_TABLE_ROWS)
        _enforce_min_rows(self.tableB, rows=DEFAULT_MIN_TABLE_ROWS)
        _enforce_min_rows(self.previewA, rows=DEFAULT_MIN_TABLE_ROWS)
        _enforce_min_rows(self.previewB, rows=DEFAULT_MIN_TABLE_ROWS)

        self.removeA_btn = QPushButton("Remove Selected (A)", self)
        self.clearA_btn = QPushButton("Clear A", self)
        self.removeB_btn = QPushButton("Remove Selected (B)", self)
        self.clearB_btn = QPushButton("Clear B", self)

        # ---------- Manual Selection (top; not inside any splitter) ----------
        manual_box = QGroupBox("Manual Selection", self)
        manual_layout = QVBoxLayout(manual_box)
        manual_layout.setContentsMargins(8, 8, 8, 8)
        manual_layout.setSpacing(8)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.addRow("Mode", self.mode_combo)
        form.addRow("Aggregator", self.agg_combo)
        manual_layout.addLayout(form)

        manual_layout.addWidget(self.targets_box)
        manual_layout.addWidget(self.armed_label)

        auto_box = QGroupBox("Auto-populate", manual_box)
        auto_form = QFormLayout(auto_box)
        auto_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        auto_form.addRow("Row", self.row_spin)
        auto_form.addRow("Col", self.col_spin)
        auto_form.addRow("Across", self.scope_combo)
        auto_form.addRow(self.autofill_btn)
        manual_layout.addWidget(auto_box)

        self.targets_box.setMinimumHeight(160)

        # --- Lists area (everything aligned vertically; single FLAT splitter) ---
        a_buttons_row = QHBoxLayout()
        a_buttons_row.addWidget(self.removeA_btn)
        a_buttons_row.addWidget(self.clearA_btn)
        a_buttons_widget = QWidget(self)
        a_buttons_widget.setLayout(a_buttons_row)

        a_picks_box = QGroupBox("Selection A — Picks", self)
        a_picks_box.setMinimumHeight(self._MIN_CARD_HEIGHT + 60)
        a_picks_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        a_picks_box.setStyleSheet("QGroupBox{font-weight: 600;}")
        a_picks_v = QVBoxLayout(a_picks_box)
        a_picks_v.setContentsMargins(8, 8, 8, 8)
        a_picks_v.setSpacing(6)
        a_picks_v.addWidget(self.tableA)
        a_picks_v.addWidget(a_buttons_widget)

        a_comp_box = QGroupBox("Computed A (per file)", self)
        a_comp_box.setMinimumHeight(self._MIN_CARD_HEIGHT)
        a_comp_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        a_comp_box.setStyleSheet("QGroupBox{font-weight: 600;}")
        a_comp_v = QVBoxLayout(a_comp_box)
        a_comp_v.setContentsMargins(8, 8, 8, 8)
        a_comp_v.setSpacing(6)
        a_comp_v.addWidget(self.previewA)

        b_buttons_row = QHBoxLayout()
        b_buttons_row.addWidget(self.removeB_btn)
        b_buttons_row.addWidget(self.clearB_btn)
        b_buttons_widget = QWidget(self)
        b_buttons_widget.setLayout(b_buttons_row)

        b_picks_box = QGroupBox("Selection B — Picks", self)
        b_picks_box.setMinimumHeight(self._MIN_CARD_HEIGHT + 60)
        b_picks_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        b_picks_box.setStyleSheet("QGroupBox{font-weight: 600;}")
        b_picks_v = QVBoxLayout(b_picks_box)
        b_picks_v.setContentsMargins(8, 8, 8, 8)
        b_picks_v.setSpacing(6)
        b_picks_v.addWidget(self.tableB)
        b_picks_v.addWidget(b_buttons_widget)

        b_comp_box = QGroupBox("Computed B (per file)", self)
        b_comp_box.setMinimumHeight(self._MIN_CARD_HEIGHT)
        b_comp_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        b_comp_box.setStyleSheet("QGroupBox{font-weight: 600;}")
        b_comp_v = QVBoxLayout(b_comp_box)
        b_comp_v.setContentsMargins(8, 8, 8, 8)
        b_comp_v.setSpacing(6)
        b_comp_v.addWidget(self.previewB)

        # --- Flattened lists splitter ---
        lists_splitter = QSplitter(Qt.Vertical, self)
        lists_splitter.setChildrenCollapsible(False)
        lists_splitter.setHandleWidth(6)
        lists_splitter.addWidget(a_picks_box)
        lists_splitter.addWidget(a_comp_box)
        lists_splitter.addWidget(b_picks_box)
        lists_splitter.addWidget(b_comp_box)
        lists_splitter.setStretchFactor(0, 4)
        lists_splitter.setStretchFactor(1, 3)
        lists_splitter.setStretchFactor(2, 4)
        lists_splitter.setStretchFactor(3, 3)
        lists_splitter.setSizes(list(self._DEFAULT_LISTS_SIZES))
        self._lists_splitter = lists_splitter
        self._default_lists_sizes = list(self._DEFAULT_LISTS_SIZES)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addWidget(manual_box)
        root.addWidget(lists_splitter, 1)

        # Wire
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.agg_combo.currentTextChanged.connect(lambda _: self._recompute_and_emit())
        self.autofill_btn.clicked.connect(self._on_autofill)
        self.removeA_btn.clicked.connect(lambda: self._remove_selected("A"))
        self.clearA_btn.clicked.connect(lambda: self._clear_bucket("A"))
        self.removeB_btn.clicked.connect(lambda: self._remove_selected("B"))
        self.clearB_btn.clicked.connect(lambda: self._clear_bucket("B"))

        # Build radios for the default mode
        self._apply_mode_visibility("Single")
        self._update_armed_ui()
        # Apply splitter sizes once we're laid out so defaults stick
        QTimer.singleShot(0, self.reset_splitters)

    # ----------------------------- Public API -----------------------------
    def set_context(self, file_to_tag: Dict[str, str]) -> None:
        self._file_to_tag = dict(file_to_tag or {})

    def reset_splitters(self) -> None:
        """Restore splitter sizes to their initial proportions."""

        if hasattr(self, "_lists_splitter") and self._lists_splitter is not None:
            self._lists_splitter.setSizes(list(self._default_lists_sizes))

    def sizeHint(self) -> QSize:  # pragma: no cover - GUI sizing hint
        height = 160 + sum(self._DEFAULT_LISTS_SIZES)
        return QSize(900, max(1200, height))

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
        self._apply_mode_visibility(requested_mode)

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
        """Switch Single / Ratio / Difference by rebuilding the relevant radios."""
        self._mode = (new_mode or "Single")
        self._apply_mode_visibility(self._mode)
        self._refresh_tables()
        self._recompute_and_emit()

    def _blank_picks(self) -> Dict[str, Dict[str, Dict[str, List[Tuple[int, int, float]]]]]:
        components = ("single", "num", "den", "left", "right")
        return {
            bucket: {comp: {} for comp in components}
            for bucket in ("A", "B")
        }

    def _target_keys_for_mode(self, mode: str) -> List[Tuple[str, str, str]]:
        mode = (mode or "Single").strip()
        if mode == "Ratio":
            return [
                ("A", "num", "A (numerator)"),
                ("A", "den", "A (denominator)"),
                ("B", "num", "B (numerator)"),
                ("B", "den", "B (denominator)"),
            ]
        if mode == "Difference":
            return [
                ("A", "left", "A (left)"),
                ("A", "right", "A (right)"),
                ("B", "left", "B (left)"),
                ("B", "right", "B (right)"),
            ]
        return [("A", "single", "A (single)"), ("B", "single", "B (single)")]

    def _rebuild_target_radios(self, mode: str) -> None:
        """Show/hide pre-created radios for the requested mode; no deletion."""
        keys = self._target_keys_for_mode(mode)
        visible_keys = {f"{bucket}.{component}" for (bucket, component, _) in keys}

        for key, rb in self._target_radio_map.items():
            rb.setVisible(key in visible_keys)

        desired = self._armed if self._armed in visible_keys else next(iter(visible_keys))
        self._armed = desired
        chosen = self._target_radio_map.get(desired)
        if chosen is not None:
            chosen.setChecked(True)
        self._update_armed_ui()

    def _apply_mode_visibility(self, mode: str) -> None:
        """Adjust radio visibility for the requested mode and keep state valid."""
        self._rebuild_target_radios(mode)

    def _on_radio_toggled(self, button: QRadioButton, checked: bool) -> None:
        """Single handler for all armed-target radios."""
        if not checked or button is None:
            return
        for key, rb in self._target_radio_map.items():
            if rb is button:
                self._armed = key
                break
        self._update_armed_ui()

    def _set_armed_radio(self, key: str) -> None:
        self._armed = key
        radio = self._target_radio_map.get(key)
        if radio is not None and not radio.isChecked():
            radio.setChecked(True)
        else:
            self._update_armed_ui()

    def _on_autofill(self) -> None:
        target = self._armed
        row1 = int(self.row_spin.value())
        col1 = int(self.col_spin.value())
        scope = self.scope_combo.currentText()  # 'Selected' or 'All'
        self.autopopulateRequested.emit(target, row1, col1, scope)
        # Button text reflects the armed target; no further updates needed here.

    def _update_armed_ui(self) -> None:
        try:
            bucket, component = self._armed.split(".", 1)
        except ValueError:
            bucket, component = "A", "single"
            self._armed = "A.single"

        component_name = {
            "single": "single",
            "num": "numerator",
            "den": "denominator",
            "left": "left",
            "right": "right",
        }.get(component, component)

        label = f"{bucket} ({component_name})"
        self.armed_label.setText(
            f"Armed target: {label}. Double-click a value in the left grid to add it."
        )
        self.autofill_btn.setText(f"Auto-populate into {label}")

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
        """Remove the highlighted rows from Selection A/B (exact pick deletion)."""
        table = self.tableA if bucket == "A" else self.tableB
        selection = table.selectionModel()
        if selection is None:
            return
        rows = sorted({index.row() for index in selection.selectedRows()})
        if not rows:
            return

        to_remove: List[Tuple[str, str, int, int, float]] = []
        for row in rows:
            file_item = table.item(row, 0)
            comp_item = table.item(row, 2)
            r_item = table.item(row, 3)
            c_item = table.item(row, 4)
            v_item = table.item(row, 5)
            if not (file_item and comp_item and r_item and c_item and v_item):
                continue
            try:
                r1 = int(r_item.text())
                c1 = int(c_item.text())
                val = float(v_item.text())
            except Exception:
                continue
            to_remove.append((file_item.text(), comp_item.text(), r1, c1, val))

        if not to_remove:
            return

        for file_id, comp, r1, c1, val in to_remove:
            comp_map = self._picks[bucket].get(comp, {})
            lst = comp_map.get(file_id, [])
            for idx, (rr, cc, vv) in enumerate(lst):
                if rr == r1 and cc == c1 and abs(vv - val) <= 1e-12:
                    del lst[idx]
                    break
            if not lst and file_id in comp_map:
                comp_map.pop(file_id, None)

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
        """Fill a compact 'file, value' table for the computed bucket."""
        if df is None or df.empty:
            table.setRowCount(0)
            return
        rows: List[Tuple[str, object, str]] = []
        for _, series in df.iterrows():
            file_id = str(series.get("file", ""))
            value = series.get("value", math.nan)
            tag = self._file_to_tag.get(file_id, "")
            rows.append((file_id, value, tag))
        table.setRowCount(len(rows))
        for idx, (file_id, value, tag) in enumerate(rows):
            table.setItem(idx, 0, QTableWidgetItem(file_id))
            if tag:
                table.item(idx, 0).setToolTip(tag)
            if pd.isna(value):
                table.setItem(idx, 1, QTableWidgetItem(""))
            else:
                try:
                    formatted = f"{float(value):.6g}"
                except Exception:
                    formatted = str(value)
                value_item = QTableWidgetItem(formatted)
                if tag:
                    value_item.setToolTip(tag)
                table.setItem(idx, 1, value_item)
        table.resizeColumnsToContents()
