"""Entry point for running the Raman Analyzer as a module."""

from PyQt5.QtWidgets import QApplication

from raman_analyzer.ui.main_window import MainWindow

import sys


def main() -> None:
    """Launch the Raman Analyzer main window."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
