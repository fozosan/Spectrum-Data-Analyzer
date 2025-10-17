from __future__ import annotations

import importlib
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication


def main() -> None:
    """
    Canonical entry point for the Raman Analyzer UI.
    Always imports the top-level package that contains SelectionPanel.
    """
    # Show which main_window module we actually imported (helps detect path issues).
    mw = importlib.import_module("raman_analyzer.ui.main_window")
    print(f"[raman] main_window module: {mw.__file__}", flush=True)

    # Lazily import after verifying import path
    from raman_analyzer.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
