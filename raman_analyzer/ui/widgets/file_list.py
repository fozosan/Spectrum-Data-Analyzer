"""Widget displaying loaded files and editable tags."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class FileListWidget(QWidget):
    """Show loaded files with editable group tags."""

    tagChanged = pyqtSignal(str, str)
    selectionChanged = pyqtSignal(list)
    xChanged = pyqtSignal(str, object)  # (file_id, float|None)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._updating = False
        self.table = QTableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File", "Tag", "X"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.MultiSelection)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._emit_selection)

    def set_files(
        self,
        files: Iterable[str],
        tags: Dict[str, str],
        x_mapping: Optional[Dict[str, float]] = None,
    ) -> None:
        """Populate the table with file/tag pairs."""

        self._updating = True
        files = list(files)
        self.table.setRowCount(len(files))
        for row, file_id in enumerate(files):
            file_item = QTableWidgetItem(file_id)
            file_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            tag_item = QTableWidgetItem(tags.get(file_id, ""))
            x_val = ""
            if x_mapping and file_id in x_mapping:
                x_val = str(x_mapping[file_id])
            x_item = QTableWidgetItem(x_val)
            self.table.setItem(row, 0, file_item)
            self.table.setItem(row, 1, tag_item)
            self.table.setItem(row, 2, x_item)
        self._updating = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        file_item = self.table.item(item.row(), 0)
        if not file_item:
            return
        file_id = file_item.text()
        if item.column() == 1:
            self.tagChanged.emit(file_id, item.text())
        elif item.column() == 2:
            text = item.text().strip()
            try:
                value = float(text) if text != "" else None
            except ValueError:
                self._updating = True
                item.setText("")
                self._updating = False
                return
            self.xChanged.emit(file_id, value)

    def _emit_selection(self) -> None:
        files = self.selected_files
        self.selectionChanged.emit(files)

    @property
    def selected_files(self) -> List[str]:
        rows = self.table.selectionModel().selectedRows()
        files = []
        for index in rows:
            file_item = self.table.item(index.row(), 0)
            if file_item:
                files.append(file_item.text())
        return files
