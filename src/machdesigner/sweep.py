"""Simulation input specifications: wavelength sweeps and Monte Carlo settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SPEED_OF_LIGHT", "MonteCarloSpec", "SweepSpec"]

#: Speed of light in vacuum, m/s (CODATA, exact by definition).
SPEED_OF_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class SweepSpec:
    """A wavelength sweep and the laser driving it.

    Parameters
    ----------
    start_m, stop_m:
        Inclusive wavelength bounds in metres.
    points:
        Number of samples. The default of 500 matches simphony's own
        resolution over the C band.
    laser_power_w:
        Optical power launched into the input coupler, in watts.
    """

    start_m: float = 1500e-9
    stop_m: float = 1600e-9
    points: int = 500
    laser_power_w: float = 20e-3

    def __post_init__(self) -> None:
        if not np.isfinite([self.start_m, self.stop_m]).all():
            raise ValueError("sweep bounds must be finite")
        if self.start_m <= 0 or self.stop_m <= 0:
            raise ValueError("wavelengths must be positive")
        if self.stop_m <= self.start_m:
            raise ValueError(
                f"stop wavelength ({self.stop_m} m) must exceed start ({self.start_m} m)"
            )
        if self.points < 2:
            raise ValueError("a sweep needs at least 2 points")
        if self.laser_power_w < 0:
            raise ValueError("laser power cannot be negative")

    @property
    def wavelengths_m(self) -> np.ndarray:
        """The sampled wavelengths, ascending."""
        return np.linspace(self.start_m, self.stop_m, self.points)

    @property
    def frequencies_hz(self) -> np.ndarray:
        """The sampled wavelengths expressed as optical frequencies."""
        return SPEED_OF_LIGHT / self.wavelengths_m

    @property
    def resolution_m(self) -> float:
        """Spacing between adjacent samples."""
        return (self.stop_m - self.start_m) / (self.points - 1)

    def resolves(self, fsr_m: float, min_points_per_fringe: int = 8) -> bool:
        """Whether this sweep samples a fringe of width ``fsr_m`` adequately.

        Used to warn when an interferometer is so unbalanced that its fringes
        fall below the sweep resolution and the plot would alias.
        """
        if fsr_m <= 0:
            return True
        return fsr_m / self.resolution_m >= min_points_per_fringe


@dataclass(frozen=True)
class MonteCarloSpec:
    """Process-variation settings for statistical simulation.

    Waveguide width and thickness variation both appear, to first order, as an
    offset in effective index; ``sigma_n_eff`` bundles them into a single
    Gaussian perturbation applied independently to each waveguide.

    The default of 2e-3 corresponds to a few nanometres of width and thickness
    variation on a 220 nm x 500 nm SOI strip waveguide, taking the sensitivity
    to be of order 1e-3 per nm of width and 2e-3 per nm of thickness. It is an
    order-of-magnitude figure chosen so the default plot shows a meaningful
    spread; calibrate it against wafer data before drawing yield conclusions.

    Parameters
    ----------
    runs:
        Number of independent samples.
    sigma_n_eff:
        Standard deviation of the per-waveguide effective-index offset.
    sigma_length_m:
        Standard deviation of per-waveguide length error. Defaults to zero,
        since lithographic length control is usually far tighter than index
        control.
    seed:
        Seed for the random generator. Fixing it makes a run reproducible,
        which is why the CLI defaults it rather than leaving it to chance.
    """

    runs: int = 10
    sigma_n_eff: float = 2e-3
    sigma_length_m: float = 0.0
    seed: int | None = 0

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError("Monte Carlo needs at least one run")
        if self.sigma_n_eff < 0 or self.sigma_length_m < 0:
            raise ValueError("standard deviations cannot be negative")

    def generator(self) -> np.random.Generator:
        """A fresh random generator honouring :attr:`seed`."""
        return np.random.default_rng(self.seed)
