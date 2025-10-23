"""Application entry point for the Raman Analyzer (Tkinter)."""
from __future__ import annotations


def main() -> None:
    """Launch the Tk-based Raman Analyzer application."""
    from raman_analyzer.tkui.app import main as tk_main

    tk_main()


if __name__ == "__main__":
    main()
