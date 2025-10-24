"""Application entry point for the Raman Analyzer (Tkinter)."""
from __future__ import annotations

import tkinter as tk

from raman_analyzer.models.session import AnalysisSession
from raman_analyzer.tkui.app import TkRamanApp


def main() -> None:
    """Launch the Tk-based Raman Analyzer application."""
    root = tk.Tk()
    session = AnalysisSession()
    TkRamanApp(root, session=session)
    root.mainloop()


if __name__ == "__main__":
    main()
