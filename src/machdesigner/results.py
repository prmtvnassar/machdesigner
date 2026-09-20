"""Simulation outputs and the analysis helpers that operate on them."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .netlist import Circuit
from .sweep import SPEED_OF_LIGHT, SweepSpec

__all__ = ["MonteCarloResult", "SweepResult"]


@dataclass(frozen=True)
class SweepResult:
    """Detected power against wavelength for one circuit realisation."""

    wavelengths_m: np.ndarray
    power_w: np.ndarray
    circuit: Circuit
    spec: SweepSpec
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.wavelengths_m.shape != self.power_w.shape:
            raise ValueError(
                f"wavelength and power arrays disagree: "
                f"{self.wavelengths_m.shape} vs {self.power_w.shape}"
            )

    # -------------------------------------------------------------- derived

    @property
    def frequencies_hz(self) -> np.ndarray:
        return SPEED_OF_LIGHT / self.wavelengths_m

    @property
    def transmission(self) -> np.ndarray:
        """Detected power normalised to the launched laser power."""
        if self.spec.laser_power_w == 0:
            return np.zeros_like(self.power_w)
        return self.power_w / self.spec.laser_power_w

    @property
    def transmission_db(self) -> np.ndarray:
        """Transmission in dB, floored to avoid ``log10(0)``."""
        floor = np.finfo(float).tiny
        return 10.0 * np.log10(np.maximum(self.transmission, floor))

    @property
    def insertion_loss_db(self) -> float:
        """Loss in dB at the best-transmitting wavelength in the sweep."""
        return float(-self.transmission_db.max())

    @property
    def extinction_ratio_db(self) -> float:
        """Ratio between the brightest and darkest points of the sweep.

        For an interferometer this is the fringe contrast. For a straight
        waveguide it reflects only the grating-coupler envelope.
        """
        return float(self.transmission_db.max() - self.transmission_db.min())

    def free_spectral_range_m(self) -> float | None:
        """Mean fringe spacing, measured from the transmission minima.

        Returns ``None`` when fewer than two minima fall inside the sweep, as
        happens for a balanced interferometer or a straight waveguide.
        """
        minima = self._minima_indices()
        if minima.size < 2:
            return None
        return float(np.mean(np.diff(self.wavelengths_m[minima])))

    def _minima_indices(self) -> np.ndarray:
        """Indices of interior local minima of the transmission curve."""
        y = self.transmission
        if y.size < 3:
            return np.empty(0, dtype=int)
        interior = np.flatnonzero((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:])) + 1
        return interior

    # --------------------------------------------------------------- export

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe summary plus the full trace."""
        fsr = self.free_spectral_range_m()
        return {
            "backend": self.backend,
            "circuit": {
                "topology": self.circuit.topology.value,
                "components": [c.label for c in self.circuit.components],
                "arm_lengths_m": list(self.circuit.arm_lengths_m),
                "path_imbalance_m": self.circuit.path_imbalance_m,
            },
            "sweep": {
                "start_m": self.spec.start_m,
                "stop_m": self.spec.stop_m,
                "points": self.spec.points,
                "laser_power_w": self.spec.laser_power_w,
            },
            "summary": {
                "insertion_loss_db": self.insertion_loss_db,
                "extinction_ratio_db": self.extinction_ratio_db,
                "free_spectral_range_m": fsr,
                "free_spectral_range_nm": None if fsr is None else fsr * 1e9,
            },
            "metadata": self.metadata,
            "trace": {
                "wavelength_m": self.wavelengths_m.tolist(),
                "power_w": self.power_w.tolist(),
            },
        }

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return path

    def write_csv(self, path: str | Path) -> Path:
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["wavelength_m", "wavelength_nm", "power_w", "transmission", "transmission_db"]
            )
            for wl, p, t, tdb in zip(
                self.wavelengths_m,
                self.power_w,
                self.transmission,
                self.transmission_db,
                strict=True,
            ):
                writer.writerow([wl, wl * 1e9, p, t, tdb])
        return path


@dataclass(frozen=True)
class MonteCarloResult:
    """An ensemble of sweeps over randomised process parameters."""

    nominal: SweepResult
    runs: tuple[SweepResult, ...]
    sigma_n_eff: float
    seed: int | None

    @property
    def power_matrix_w(self) -> np.ndarray:
        """Stacked run powers, shape ``(runs, points)``."""
        return np.vstack([r.power_w for r in self.runs])

    @property
    def transmission_matrix(self) -> np.ndarray:
        return np.vstack([r.transmission for r in self.runs])

    @property
    def mean_transmission(self) -> np.ndarray:
        return self.transmission_matrix.mean(axis=0)

    @property
    def std_transmission(self) -> np.ndarray:
        return (
            self.transmission_matrix.std(axis=0, ddof=1)
            if len(self.runs) > 1
            else np.zeros_like(self.mean_transmission)
        )

    def percentile_band(
        self, lower: float = 5.0, upper: float = 95.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise percentile envelope across the ensemble."""
        matrix = self.transmission_matrix
        return (
            np.percentile(matrix, lower, axis=0),
            np.percentile(matrix, upper, axis=0),
        )

    def fsr_statistics_m(self) -> dict[str, float] | None:
        """Mean and spread of the measured free spectral range across runs."""
        values = [r.free_spectral_range_m() for r in self.runs]
        usable = [v for v in values if v is not None]
        if len(usable) < 2:
            return None
        array = np.asarray(usable, dtype=float)
        return {
            "mean_m": float(array.mean()),
            "std_m": float(array.std(ddof=1)),
            "min_m": float(array.min()),
            "max_m": float(array.max()),
            "samples": float(array.size),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "nominal": self.nominal.to_dict(),
            "monte_carlo": {
                "runs": len(self.runs),
                "sigma_n_eff": self.sigma_n_eff,
                "seed": self.seed,
                "fsr_statistics_m": self.fsr_statistics_m(),
                "traces": [r.power_w.tolist() for r in self.runs],
            },
        }

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=indent), encoding="utf-8")
        return path
