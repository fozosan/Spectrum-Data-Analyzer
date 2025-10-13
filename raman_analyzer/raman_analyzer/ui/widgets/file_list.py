"""Widget displaying loaded files and editable tags."""
from __future__ import annotations

from typing import Dict, Iterable, List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


class FileListWidget(QWidget):
    """Show loaded files with editable group tags."""

    tagChanged = pyqtSignal(str, str)
    selectionChanged = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._updating = False
        self.table = QTableWidget(self)
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["File", "Tag"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.MultiSelection)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

        self.table.itemChanged.connect(self._on_item_changed)
        self.table.itemSelectionChanged.connect(self._emit_selection)

    def set_files(self, files: Iterable[str], tags: Dict[str, str]) -> None:
        """Populate the table with file/tag pairs."""

        self._updating = True
        files = list(files)
        self.table.setRowCount(len(files))
        for row, file_id in enumerate(files):
            file_item = QTableWidgetItem(file_id)
            file_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            tag_item = QTableWidgetItem(tags.get(file_id, ""))
            self.table.setItem(row, 0, file_item)
            self.table.setItem(row, 1, tag_item)
        self._updating = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != 1:
            return
        file_item = self.table.item(item.row(), 0)
        if file_item is None:
            return
        file_id = file_item.text()
        tag = item.text()
        self.tagChanged.emit(file_id, tag)

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
