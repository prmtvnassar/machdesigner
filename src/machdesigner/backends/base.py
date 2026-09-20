"""Backend protocol shared by the analytic and simphony engines."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..netlist import Circuit
from ..results import MonteCarloResult, SweepResult
from ..sweep import MonteCarloSpec, SweepSpec

__all__ = ["Backend", "BackendUnavailableError"]


class BackendUnavailableError(RuntimeError):
    """Raised when a backend's optional dependencies are missing.

    The message explains how to install what is needed, so a user who asks for
    the simphony backend without simphony installed gets an actionable error
    rather than an ``ImportError`` traceback.
    """


@runtime_checkable
class Backend(Protocol):
    """Anything that can turn a :class:`Circuit` into a :class:`SweepResult`."""

    name: str

    def sweep(self, circuit: Circuit, spec: SweepSpec) -> SweepResult:
        """Simulate a single wavelength sweep."""
        ...

    def monte_carlo(
        self, circuit: Circuit, spec: SweepSpec, mc: MonteCarloSpec
    ) -> MonteCarloResult:
        """Simulate an ensemble of sweeps over randomised parameters."""
        ...
