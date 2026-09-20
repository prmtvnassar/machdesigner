"""Embedded matplotlib canvas.

Plots are drawn into a widget inside the main window rather than into separate
pyplot windows. Besides fixing the blank-thumbnail bug, this means the figure
stays interactive: the navigation toolbar gives pan, zoom, and a working
"save figure" button.
"""

from __future__ import annotations

from matplotlib.figure import Figure
from PyQt5.QtWidgets import QVBoxLayout, QWidget

try:  # matplotlib >= 3.5 ships a binding-agnostic Qt backend
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )
except ImportError:  # pragma: no cover - older matplotlib
    from matplotlib.backends.backend_qt5agg import (  # type: ignore[no-redef]
        FigureCanvasQTAgg,
        NavigationToolbar2QT,
    )

__all__ = ["PlotCanvas"]


class PlotCanvas(QWidget):
    """A figure, its axes, and a navigation toolbar."""

    def __init__(self, parent: QWidget | None = None, *, toolbar: bool = True) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(5.5, 3.6), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

        self.show_placeholder()

    def clear(self) -> None:
        """Reset the axes.

        Called before every redraw. The original never cleared, so each run
        layered new curves on top of the previous ones.
        """
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)

    def show_placeholder(self, message: str = "Press Run to simulate") -> None:
        self.clear()
        self.axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.axes.transAxes,
            fontsize=10,
            color="0.45",
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        for spine in self.axes.spines.values():
            spine.set_visible(False)
        self.draw()

    def draw(self) -> None:
        self.canvas.draw_idle()

    def save(self, path: str, dpi: int = 200) -> None:
        self.figure.savefig(path, dpi=dpi)
