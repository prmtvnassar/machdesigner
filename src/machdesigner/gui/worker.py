"""Background simulation worker.

The original GUI ran simulations directly in the button handler and called
``plt.show()``, so the window froze for the duration and then blocked until the
user dismissed a separate plot window. Here the work happens on a QThread and
results come back by signal, leaving the event loop responsive.
"""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal

from ..netlist import Circuit
from ..results import MonteCarloResult, SweepResult
from ..simulate import run_monte_carlo, simulate
from ..sweep import MonteCarloSpec, SweepSpec

__all__ = ["SimulationWorker"]


class SimulationWorker(QThread):
    """Run a sweep, and optionally a Monte Carlo study, off the UI thread.

    Signals
    -------
    finished_sweep:
        Emitted with the nominal :class:`SweepResult`.
    finished_monte_carlo:
        Emitted with a :class:`MonteCarloResult`, only when requested.
    failed:
        Emitted with a human-readable message if anything raised.
    progress:
        Emitted with short status strings for the status bar.
    """

    finished_sweep = pyqtSignal(object)
    finished_monte_carlo = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        circuit: Circuit,
        spec: SweepSpec,
        backend: str = "auto",
        monte_carlo: MonteCarloSpec | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._circuit = circuit
        self._spec = spec
        self._backend = backend
        self._mc = monte_carlo

    def run(self) -> None:
        try:
            self.progress.emit(f"Running sweep on the {self._backend} backend...")
            result: SweepResult = simulate(self._circuit, self._spec, backend=self._backend)
            self.finished_sweep.emit(result)

            if self._mc is not None:
                self.progress.emit(f"Running {self._mc.runs} Monte Carlo samples...")
                mc_result: MonteCarloResult = run_monte_carlo(
                    self._circuit, self._spec, self._mc, backend=self._backend
                )
                self.finished_monte_carlo.emit(mc_result)

            self.progress.emit("Done.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
