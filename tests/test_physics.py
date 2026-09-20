"""Validate the analytic backend against closed-form interferometer physics.

These are the tests that matter: they check simulated output against
independently derived relations rather than against a previously recorded
output, so a regression in the model shows up as a physics failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from machdesigner import Circuit, SweepSpec, simulate
from machdesigner.backends.analytic import AnalyticBackend
from machdesigner.components import WaveguideModel

# --------------------------------------------------------------- waveguide


def test_group_index_matches_numerical_derivative():
    """n_g = n_eff - lambda dn_eff/dlambda, checked against finite differences."""
    wg = WaveguideModel()
    lam = np.linspace(1.50e-6, 1.60e-6, 21)
    h = 1e-11
    dn_dlam = (wg.n_eff(lam + h) - wg.n_eff(lam - h)) / (2 * h)
    expected = wg.n_eff(lam) - lam * dn_dlam
    np.testing.assert_allclose(wg.n_group(lam), expected, rtol=1e-6)


def test_group_index_is_physical_for_soi_strip():
    """A 220x500 nm SOI strip guide has n_g near 4.2 at 1550 nm."""
    n_g = float(WaveguideModel().n_group(1.55e-6))
    assert 3.9 < n_g < 4.5


def test_waveguide_loss_follows_beer_lambert():
    """Power should fall by exactly alpha * L dB over length L."""
    wg = WaveguideModel(loss_db_per_cm=2.4)
    length_m = 0.01  # 1 cm
    amplitude = wg.amplitude(1.55e-6, length_m)
    power_db = 20 * np.log10(np.abs(amplitude))
    assert power_db == pytest.approx(-2.4, abs=1e-9)


# --------------------------------------------------------------------- MZI


def test_ideal_mzi_follows_cosine_squared_law(ideal_pdk):
    """A lossless MZI must reproduce T = cos^2(pi n_eff dL / lambda)."""
    long_m, short_m = 150e-6, 50e-6
    circuit = Circuit.mzi(long_arm_m=long_m, short_arm_m=short_m)
    spec = SweepSpec(points=3000)
    result = AnalyticBackend(ideal_pdk).sweep(circuit, spec)

    lam = result.wavelengths_m
    n_eff = ideal_pdk.waveguide.n_eff(lam)
    expected = np.cos(np.pi * n_eff * (long_m - short_m) / lam) ** 2

    np.testing.assert_allclose(result.transmission, expected, atol=1e-12)


def test_ideal_mzi_reaches_full_transmission_and_full_extinction(ideal_pdk):
    """Fringe peaks hit unity and nulls reach zero when nothing is lossy."""
    circuit = Circuit.mzi(long_arm_m=150e-6, short_arm_m=50e-6)
    result = AnalyticBackend(ideal_pdk).sweep(circuit, SweepSpec(points=20000))
    assert result.transmission.max() == pytest.approx(1.0, abs=1e-6)
    assert result.transmission.min() < 1e-6


def test_transmission_never_exceeds_unity(default_pdk):
    """Energy conservation: a passive circuit cannot amplify."""
    for long_m, short_m in [(150e-6, 50e-6), (500e-6, 10e-6), (80e-6, 80e-6)]:
        circuit = Circuit.mzi(long_arm_m=long_m, short_arm_m=short_m)
        result = AnalyticBackend(default_pdk).sweep(circuit, SweepSpec(points=2000))
        assert result.transmission.max() <= 1.0 + 1e-12


def test_free_spectral_range_matches_theory():
    """Measured fringe spacing must match lambda^2 / (n_g dL)."""
    circuit = Circuit.mzi(long_arm_m=150e-6, short_arm_m=50e-6)
    result = simulate(circuit, SweepSpec(points=20000))

    measured = result.free_spectral_range_m()
    predicted = result.metadata["predicted_fsr_m"]
    assert measured is not None
    # The prediction uses n_g at 1550 nm while the sweep spans 1500-1600 nm,
    # over which the FSR drifts by a few percent; 3% covers that drift.
    assert measured == pytest.approx(predicted, rel=0.03)


@pytest.mark.parametrize("imbalance_um", [50.0, 100.0, 200.0, 400.0])
def test_fsr_scales_inversely_with_imbalance(imbalance_um):
    """Doubling the arm imbalance halves the free spectral range."""
    short_m = 50e-6
    circuit = Circuit.mzi(long_arm_m=short_m + imbalance_um * 1e-6, short_arm_m=short_m)
    # Keep roughly constant points-per-fringe as fringes get narrower.
    result = simulate(circuit, SweepSpec(points=int(200 * imbalance_um)))
    measured = result.free_spectral_range_m()

    n_g = 4.1765  # default SOI strip guide at 1550 nm
    expected = 1.55e-6**2 / (n_g * imbalance_um * 1e-6)
    assert measured == pytest.approx(expected, rel=0.05)


def test_balanced_mzi_has_no_fringes(default_pdk):
    """Equal arms interfere constructively everywhere, so there is no ripple."""
    circuit = Circuit.mzi(long_arm_m=100e-6, short_arm_m=100e-6)
    result = AnalyticBackend(default_pdk).sweep(circuit, SweepSpec(points=2000))
    assert circuit.path_imbalance_m == 0.0
    assert result.free_spectral_range_m() is None
    # The only structure left is the grating-coupler envelope, which is smooth
    # and monotonic on each side of its peak.
    assert np.all(np.diff(result.transmission[result.transmission.argmax() :]) <= 1e-12)


def test_balanced_mzi_equals_straight_guide_up_to_splitter_loss(ideal_pdk):
    """With ideal splitters, a balanced MZI transmits like a plain guide."""
    length_m = 100e-6
    spec = SweepSpec(points=500)
    backend = AnalyticBackend(ideal_pdk)
    mzi = backend.sweep(Circuit.mzi(length_m, length_m), spec)
    straight = backend.sweep(Circuit.straight(length_m), spec)
    np.testing.assert_allclose(mzi.transmission, straight.transmission, atol=1e-12)


# ---------------------------------------------------------------- straight


def test_straight_guide_loss_is_couplers_plus_propagation(default_pdk):
    """Peak loss = 2x coupler insertion loss + alpha * L."""
    length_m = 100e-6
    result = AnalyticBackend(default_pdk).sweep(
        Circuit.straight(length_m), SweepSpec(points=2001)
    )
    expected_db = (
        2 * default_pdk.grating_coupler.peak_insertion_loss_db
        + default_pdk.waveguide.loss_db_per_cm * length_m * 100
    )
    assert result.insertion_loss_db == pytest.approx(expected_db, abs=1e-3)


def test_grating_coupler_bandwidth_is_three_db_down(default_pdk):
    """Transmission at the stated 3 dB bandwidth edge is half the peak."""
    gc = default_pdk.grating_coupler
    peak = gc.power(gc.center_m)
    edge = gc.power(gc.center_m + gc.bandwidth_3db_m / 2)
    assert edge / peak == pytest.approx(0.5, rel=1e-9)


# ------------------------------------------------------------- monte carlo


def test_monte_carlo_is_reproducible_under_a_fixed_seed():
    from machdesigner import MonteCarloSpec, run_monte_carlo

    circuit = Circuit.mzi(150e-6, 50e-6)
    spec = SweepSpec(points=800)
    first = run_monte_carlo(circuit, spec, MonteCarloSpec(runs=5, seed=42))
    second = run_monte_carlo(circuit, spec, MonteCarloSpec(runs=5, seed=42))
    np.testing.assert_array_equal(first.power_matrix_w, second.power_matrix_w)


def test_monte_carlo_with_zero_sigma_reproduces_the_nominal_circuit():
    from machdesigner import MonteCarloSpec, run_monte_carlo

    circuit = Circuit.mzi(150e-6, 50e-6)
    spec = SweepSpec(points=800)
    result = run_monte_carlo(
        circuit, spec, MonteCarloSpec(runs=3, sigma_n_eff=0.0, sigma_length_m=0.0)
    )
    for run in result.runs:
        np.testing.assert_allclose(run.power_w, result.nominal.power_w, atol=1e-18)


def test_monte_carlo_spread_grows_with_sigma():
    from machdesigner import MonteCarloSpec, run_monte_carlo

    circuit = Circuit.mzi(150e-6, 50e-6)
    spec = SweepSpec(points=1500)
    tight = run_monte_carlo(circuit, spec, MonteCarloSpec(runs=40, sigma_n_eff=5e-4, seed=1))
    loose = run_monte_carlo(circuit, spec, MonteCarloSpec(runs=40, sigma_n_eff=5e-3, seed=1))
    assert loose.std_transmission.mean() > tight.std_transmission.mean()
