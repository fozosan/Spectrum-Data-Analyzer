"""Tabular view for pandas DataFrames."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt5.QtWidgets import QTableView, QVBoxLayout, QWidget


class DataFrameModel(QAbstractTableModel):
    """Qt model representing a pandas DataFrame."""

    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None) -> None:
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def rowCount(self, parent: Optional[QModelIndex] = None) -> int:  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        return len(self._df.index)

    def columnCount(self, parent: Optional[QModelIndex] = None) -> int:  # type: ignore[override]
        if parent and parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):  # type: ignore[override]
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        if pd.isna(value):
            return ""
        return str(value)

    def headerData(  # type: ignore[override]
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section]) if section < len(self._df.columns) else ""
        return str(self._df.index[section]) if section < len(self._df.index) else ""


class DataTableWidget(QWidget):
    """Simple wrapper displaying a DataFrame in a QTableView."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.model = DataFrameModel()
        self.view = QTableView(self)
        self.view.setModel(self.model)
        self.view.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.view)

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.model.set_dataframe(df)
