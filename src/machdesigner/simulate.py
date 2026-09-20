"""High-level simulation entry points.

These wrap the backends with the checks that are easy to forget: that the
sweep actually resolves the interference fringes, and that the chosen backend
is available.
"""

from __future__ import annotations

import warnings

from .backends import Backend, get_backend
from .netlist import Circuit, Topology
from .results import MonteCarloResult, SweepResult
from .sweep import MonteCarloSpec, SweepSpec

__all__ = ["UnderSampledSweepWarning", "run_monte_carlo", "simulate"]


class UnderSampledSweepWarning(UserWarning):
    """The sweep is too coarse to resolve the circuit's fringes.

    Emitted when the predicted free spectral range spans fewer than a handful
    of sample points. The simulation still runs, but the trace will alias and
    the plotted fringes will be misleading.
    """


def _resolve(backend: Backend | str | None) -> Backend:
    if backend is None or isinstance(backend, str):
        return get_backend(backend or "auto")
    return backend


def _warn_if_undersampled(circuit: Circuit, spec: SweepSpec) -> None:
    if circuit.topology is not Topology.MZI:
        return
    imbalance = circuit.path_imbalance_m
    if imbalance <= 0:
        return
    # lambda^2 / (n_g * dL), with a representative group index of 4.18 for the
    # default SOI strip waveguide. Only used to decide whether to warn.
    centre = 0.5 * (spec.start_m + spec.stop_m)
    fsr = centre**2 / (4.18 * imbalance)
    if not spec.resolves(fsr):
        warnings.warn(
            f"the sweep samples every {spec.resolution_m * 1e12:.1f} pm but the "
            f"fringe spacing is only about {fsr * 1e12:.1f} pm; the result will "
            f"alias. Increase 'points' or reduce the arm imbalance "
            f"({imbalance * 1e6:g} um).",
            UnderSampledSweepWarning,
            stacklevel=3,
        )


def simulate(
    circuit: Circuit,
    spec: SweepSpec | None = None,
    backend: Backend | str | None = None,
) -> SweepResult:
    """Run one wavelength sweep.

    Parameters
    ----------
    circuit:
        A validated circuit, e.g. from :meth:`Circuit.mzi`.
    spec:
        Sweep settings; defaults to 1500-1600 nm at 20 mW.
    backend:
        A backend instance, a name (``"analytic"``, ``"simphony"``,
        ``"auto"``), or ``None`` for the default.
    """
    spec = spec or SweepSpec()
    _warn_if_undersampled(circuit, spec)
    return _resolve(backend).sweep(circuit, spec)


def run_monte_carlo(
    circuit: Circuit,
    spec: SweepSpec | None = None,
    mc: MonteCarloSpec | None = None,
    backend: Backend | str | None = None,
) -> MonteCarloResult:
    """Run an ensemble of sweeps over randomised process parameters."""
    spec = spec or SweepSpec()
    mc = mc or MonteCarloSpec()
    _warn_if_undersampled(circuit, spec)
    return _resolve(backend).monte_carlo(circuit, spec, mc)
