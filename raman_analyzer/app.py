from __future__ import annotations

import sys
import importlib
from pathlib import Path

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt


def _import_main_window():
    """
    Import raman_analyzer.ui.main_window whether the app is launched as
    `python -m raman_analyzer` (package mode) or by executing app.py directly
    (script mode inside the package directory).
    """
    try:
        # Normal case: package import
        return importlib.import_module("raman_analyzer.ui.main_window")
    except ModuleNotFoundError:
        # Script launched from within the package dir: add parent of this file to sys.path
        pkg_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(pkg_dir.parent))
        return importlib.import_module("raman_analyzer.ui.main_window")


def main() -> None:
    mw_mod = _import_main_window()
    MainWindow = getattr(mw_mod, "MainWindow")

    # Basic Qt app bootstrap
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
