"""Qt front end for MachDesigner.

Importing this package requires PyQt5; the rest of the library does not. Use
``pip install 'machdesigner[gui]'`` to pull it in, or just use the CLI.
"""

from __future__ import annotations

import sys

__all__ = ["MainWindow", "launch"]


def __getattr__(name: str):
    """Import the window lazily so ``import machdesigner.gui`` stays cheap."""
    if name == "MainWindow":
        from .main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def launch(argv: list[str] | None = None) -> int:
    """Start the designer. Returns the Qt exit code.

    Reuses an existing ``QApplication`` when one is already running, so this is
    safe to call from an interactive session or a test harness.
    """
    from PyQt5.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QApplication(argv if argv is not None else sys.argv)

    window = MainWindow()
    window.show()

    if not owns_app:
        # Someone else owns the event loop; just show the window.
        return 0
    return app.exec_()
