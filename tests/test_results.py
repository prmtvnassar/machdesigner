"""Result metrics, export formats, and sweep-specification validation."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from machdesigner import (
    Circuit,
    MonteCarloSpec,
    SweepSpec,
    UnderSampledSweepWarning,
    run_monte_carlo,
    simulate,
)


@pytest.fixture
def result():
    return simulate(Circuit.mzi(150e-6, 50e-6), SweepSpec(points=2000))


# ------------------------------------------------------------- sweep specs


def test_sweep_bounds_must_be_ordered():
    with pytest.raises(ValueError, match="must exceed start"):
        SweepSpec(start_m=1600e-9, stop_m=1500e-9)


def test_sweep_needs_at_least_two_points():
    with pytest.raises(ValueError, match="at least 2 points"):
        SweepSpec(points=1)


def test_negative_power_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        SweepSpec(laser_power_w=-1.0)


def test_wavelengths_are_ascending_and_within_bounds():
    spec = SweepSpec(start_m=1500e-9, stop_m=1600e-9, points=101)
    wl = spec.wavelengths_m
    assert wl[0] == pytest.approx(1500e-9)
    assert wl[-1] == pytest.approx(1600e-9)
    assert np.all(np.diff(wl) > 0)
    assert spec.resolution_m == pytest.approx(1e-9)


def test_monte_carlo_needs_at_least_one_run():
    with pytest.raises(ValueError, match="at least one run"):
        MonteCarloSpec(runs=0)


# ----------------------------------------------------------------- metrics


def test_transmission_is_power_normalised_by_launch_power(result):
    np.testing.assert_allclose(result.transmission, result.power_w / result.spec.laser_power_w)


def test_zero_launch_power_gives_zero_transmission_not_nan():
    result = simulate(Circuit.mzi(150e-6, 50e-6), SweepSpec(points=4000, laser_power_w=0.0))
    assert np.all(result.transmission == 0.0)
    assert not np.isnan(result.transmission_db).any()


def test_transmission_db_has_no_infinities_at_nulls(result):
    """log10(0) at a perfect null must be floored, not infinite."""
    assert np.isfinite(result.transmission_db).all()


def test_insertion_loss_and_extinction_are_consistent(result):
    assert result.insertion_loss_db == pytest.approx(-result.transmission_db.max())
    assert result.extinction_ratio_db >= 0


def test_wavelength_and_frequency_axes_agree(result):
    from machdesigner.sweep import SPEED_OF_LIGHT

    np.testing.assert_allclose(result.frequencies_hz, SPEED_OF_LIGHT / result.wavelengths_m)


# ------------------------------------------------------------------ export


def test_csv_round_trips_the_trace(result, tmp_path):
    path = result.write_csv(tmp_path / "trace.csv")
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == result.wavelengths_m.size
    assert float(rows[0]["wavelength_m"]) == pytest.approx(result.wavelengths_m[0])
    assert float(rows[-1]["power_w"]) == pytest.approx(result.power_w[-1])


def test_json_is_valid_and_carries_the_summary(result, tmp_path):
    path = result.write_json(tmp_path / "result.json")
    payload = json.loads(path.read_text())
    assert payload["backend"] == "analytic"
    assert payload["circuit"]["topology"] == "mzi"
    assert payload["summary"]["free_spectral_range_nm"] == pytest.approx(
        result.free_spectral_range_m() * 1e9
    )
    assert len(payload["trace"]["power_w"]) == result.power_w.size


def test_monte_carlo_json_includes_every_run(tmp_path):
    mc = run_monte_carlo(
        Circuit.mzi(150e-6, 50e-6), SweepSpec(points=200), MonteCarloSpec(runs=4, seed=3)
    )
    payload = json.loads(mc.write_json(tmp_path / "mc.json").read_text())
    assert payload["monte_carlo"]["runs"] == 4
    assert len(payload["monte_carlo"]["traces"]) == 4
    assert payload["monte_carlo"]["seed"] == 3


def test_monte_carlo_percentile_band_brackets_the_ensemble():
    mc = run_monte_carlo(
        Circuit.mzi(150e-6, 50e-6), SweepSpec(points=400), MonteCarloSpec(runs=25, seed=5)
    )
    low, high = mc.percentile_band()
    assert np.all(low <= high)
    assert low.shape == mc.nominal.transmission.shape


# ---------------------------------------------------------------- warnings


def test_undersampled_sweep_warns():
    """A large imbalance on a coarse sweep aliases; the user should be told."""
    with pytest.warns(UnderSampledSweepWarning, match="alias"):
        simulate(Circuit.mzi(5000e-6, 50e-6), SweepSpec(points=200))


def test_adequately_sampled_sweep_does_not_warn(recwarn):
    simulate(Circuit.mzi(150e-6, 50e-6), SweepSpec(points=4000))
    assert not [w for w in recwarn if issubclass(w.category, UnderSampledSweepWarning)]


def test_straight_circuit_never_warns_about_fringes(recwarn):
    simulate(Circuit.straight(5000e-6), SweepSpec(points=50))
    assert not [w for w in recwarn if issubclass(w.category, UnderSampledSweepWarning)]
