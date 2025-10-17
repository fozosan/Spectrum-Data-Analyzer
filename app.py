"""Application entry point for the Raman Analyzer desktop tool."""
from __future__ import annotations

import importlib
import importlib.util
import sys

from PyQt5.QtWidgets import QApplication

from raman_analyzer.ui.main_window import MainWindow


def main() -> None:
    """Launch the Raman Analyzer application."""
    # Sanity check the module import path to avoid duplicate-package confusion.
    try:
        spec = importlib.util.find_spec("raman_analyzer.ui.main_window")
        print("[raman_analyzer] main_window spec:", spec)
        if spec and getattr(spec, "origin", None):
            print("[raman_analyzer] main_window origin:", spec.origin)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print("[raman_analyzer] failed to resolve main_window:", exc)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    print("[raman_analyzer] MainWindow constructed OK")
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
