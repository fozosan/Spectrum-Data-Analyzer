"""Application entry point for the Raman Analyzer desktop tool."""
from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from raman_analyzer.ui.main_window import MainWindow


def main() -> None:
    """Launch the Raman Analyzer application."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
