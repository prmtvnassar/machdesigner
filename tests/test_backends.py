"""Backend registry, and cross-validation between the two engines.

The simphony tests skip cleanly when simphony is not installed, so the suite
is meaningful in both a minimal and a full environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from machdesigner import Circuit, SweepSpec, simulate
from machdesigner.backends import (
    AnalyticBackend,
    BackendUnavailableError,
    available_backends,
    get_backend,
    simphony_available,
)

requires_simphony = pytest.mark.skipif(
    not simphony_available(), reason="simphony 0.6.x is not installed"
)


# ---------------------------------------------------------------- registry


def test_auto_resolves_to_analytic():
    assert isinstance(get_backend("auto"), AnalyticBackend)
    assert isinstance(get_backend(), AnalyticBackend)


def test_backend_names_are_case_insensitive():
    assert isinstance(get_backend("ANALYTIC"), AnalyticBackend)


def test_unknown_backend_names_are_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("quantum-annealer")


def test_analytic_is_always_available():
    assert "analytic" in available_backends()


def test_simulate_accepts_a_backend_instance():
    backend = AnalyticBackend()
    result = simulate(Circuit.straight(100e-6), SweepSpec(points=100), backend=backend)
    assert result.backend == "analytic"


# ---------------------------------------------------------------- simphony


@requires_simphony
def test_simphony_backend_runs_an_mzi():
    result = simulate(Circuit.mzi(150e-6, 50e-6), SweepSpec(), backend="simphony")
    assert result.backend == "simphony"
    assert result.power_w.size > 0
    assert np.all(np.isfinite(result.power_w))


@requires_simphony
def test_simphony_returns_ascending_wavelengths():
    """simphony works in frequency; results must come back wavelength-ordered."""
    result = simulate(Circuit.mzi(150e-6, 50e-6), SweepSpec(), backend="simphony")
    assert np.all(np.diff(result.wavelengths_m) > 0)


@requires_simphony
def test_simphony_runs_a_straight_guide():
    result = simulate(Circuit.straight(100e-6), SweepSpec(), backend="simphony")
    assert np.all(np.isfinite(result.power_w))
    assert result.free_spectral_range_m() is None


@requires_simphony
@pytest.mark.parametrize("imbalance_um", [100.0, 200.0])
def test_backends_agree_on_free_spectral_range(imbalance_um):
    """The two engines are independent; their fringe spacing must still match.

    Absolute transmission differs because the loss models differ, but the FSR
    depends only on the group index, which both engines derive from the same
    physical waveguide.
    """
    short_m = 50e-6
    circuit = Circuit.mzi(short_m + imbalance_um * 1e-6, short_m)
    spec = SweepSpec(points=20000)

    analytic = simulate(circuit, spec, backend="analytic").free_spectral_range_m()
    simphony = simulate(circuit, spec, backend="simphony").free_spectral_range_m()

    assert analytic is not None and simphony is not None
    assert analytic == pytest.approx(simphony, rel=0.05)


@requires_simphony
def test_simphony_respects_energy_conservation():
    result = simulate(Circuit.mzi(150e-6, 50e-6), SweepSpec(), backend="simphony")
    assert result.transmission.max() <= 1.0


@requires_simphony
def test_simphony_monte_carlo_produces_the_requested_runs():
    from machdesigner import MonteCarloSpec, run_monte_carlo

    result = run_monte_carlo(
        Circuit.mzi(150e-6, 50e-6), SweepSpec(), MonteCarloSpec(runs=3), backend="simphony"
    )
    assert len(result.runs) == 3


def test_missing_simphony_raises_a_helpful_error(monkeypatch):
    """Without simphony the error should say how to install it."""
    import machdesigner.backends.simphony_backend as module

    def boom():
        raise BackendUnavailableError(module._INSTALL_HINT)

    monkeypatch.setattr(module, "_import_simphony", boom)
    with pytest.raises(BackendUnavailableError, match="pip install"):
        module.SimphonyBackend()
