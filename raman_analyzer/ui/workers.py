"""Background worker objects used by the UI layer."""
from __future__ import annotations

import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from raman_analyzer.io.loader import load_csvs


class CsvLoaderWorker(QObject):
    """Load CSV files in a background thread and emit the resulting DataFrame."""

    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        self._paths = paths

    @pyqtSlot()
    def run(self) -> None:
        try:
            df = load_csvs(self._paths)
        except Exception as exc:  # pragma: no cover - defensive
            self.error.emit(str(exc))
            self.finished.emit(pd.DataFrame())
            return
        self.finished.emit(df)
