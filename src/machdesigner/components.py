"""Photonic component types and their compact models.

This module defines the building blocks of a circuit and the closed-form
("compact") models that describe how each one transforms an optical field.

All models are *nominal* silicon-on-insulator (SOI) strip-waveguide models at
telecom wavelengths. They are intended to be physically representative, not
foundry-accurate: calibrate the coefficients against a real process design kit
(PDK) before using results for tape-out decisions. Every parameter is
overridable, so recalibration means constructing a different model instance
rather than editing this file.

Conventions
-----------
* Wavelengths are in metres everywhere in the public API.
* ``amplitude()`` methods return a *field* (amplitude) transfer coefficient,
  possibly complex. Power transmission is ``abs(amplitude) ** 2``.
* Losses are quoted in dB of **power**, so an amplitude factor is
  ``10 ** (-loss_db / 20)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum

import numpy as np

__all__ = [
    "PDK",
    "Component",
    "ComponentType",
    "GratingCouplerModel",
    "WaveguideModel",
    "YBranchModel",
    "db_to_amplitude",
    "db_to_power",
]


def db_to_amplitude(loss_db: float | np.ndarray) -> float | np.ndarray:
    """Convert a power loss in dB to a field-amplitude multiplier."""
    return 10.0 ** (-np.asarray(loss_db, dtype=float) / 20.0)


def db_to_power(loss_db: float | np.ndarray) -> float | np.ndarray:
    """Convert a power loss in dB to a power multiplier."""
    return 10.0 ** (-np.asarray(loss_db, dtype=float) / 10.0)


class ComponentType(IntEnum):
    """The kinds of component a circuit slot can hold.

    The integer values intentionally match the dropdown indices used by the
    original MachDesigner GUI, so legacy circuit descriptions keep working.
    """

    NONE = 0
    GC_INPUT = 1
    GC_OUTPUT = 2
    Y_SPLITTER = 3
    Y_COMBINER = 4
    WAVEGUIDE = 5

    @property
    def label(self) -> str:
        """Human-readable name, as shown in the GUI."""
        return {
            ComponentType.NONE: "None",
            ComponentType.GC_INPUT: "Grating coupler (input)",
            ComponentType.GC_OUTPUT: "Grating coupler (output)",
            ComponentType.Y_SPLITTER: "Y-splitter",
            ComponentType.Y_COMBINER: "Y-combiner",
            ComponentType.WAVEGUIDE: "Waveguide",
        }[self]

    @classmethod
    def from_label(cls, label: str) -> ComponentType:
        """Look a type up by label or short name, case-insensitively."""
        key = label.strip().lower().replace("_", "-").replace(" ", "-")
        aliases = {
            "none": cls.NONE,
            "gc-input": cls.GC_INPUT,
            "gc-in": cls.GC_INPUT,
            "grating-coupler-(input)": cls.GC_INPUT,
            "gc-output": cls.GC_OUTPUT,
            "gc-out": cls.GC_OUTPUT,
            "grating-coupler-(output)": cls.GC_OUTPUT,
            "y-splitter": cls.Y_SPLITTER,
            "splitter": cls.Y_SPLITTER,
            "y-combiner": cls.Y_COMBINER,
            "combiner": cls.Y_COMBINER,
            "waveguide": cls.WAVEGUIDE,
            "wg": cls.WAVEGUIDE,
        }
        try:
            return aliases[key]
        except KeyError:
            raise ValueError(
                f"unknown component {label!r}; expected one of "
                + ", ".join(sorted({t.name.lower() for t in cls}))
            ) from None


@dataclass(frozen=True)
class WaveguideModel:
    """Dispersive, lossy single-mode strip waveguide.

    The effective index is a second-order Taylor expansion about
    ``lambda0_m``::

        n_eff(lam) = c0 + c1 * (lam - lam0) + c2 * (lam - lam0) ** 2

    with the wavelength offset expressed in micrometres, which is the form
    used by the SiEPIC compact models. The default coefficients describe a
    220 nm x 500 nm SOI strip waveguide in the fundamental TE mode and give a
    group index of about 4.18 at 1550 nm.
    """

    n_eff_coeffs: tuple[float, float, float] = (2.4379, -1.1217, -0.04565)
    lambda0_m: float = 1.55e-6
    loss_db_per_cm: float = 2.4

    def n_eff(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        """Effective index at the given wavelength(s)."""
        c0, c1, c2 = self.n_eff_coeffs
        x = (np.asarray(wavelength_m, dtype=float) - self.lambda0_m) * 1e6
        return c0 + c1 * x + c2 * x * x

    def n_group(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        """Group index, ``n_g = n_eff - lambda * dn_eff/dlambda``."""
        _, c1, c2 = self.n_eff_coeffs
        wl = np.asarray(wavelength_m, dtype=float)
        x = (wl - self.lambda0_m) * 1e6
        dn_dlambda_per_um = c1 + 2.0 * c2 * x
        return self.n_eff(wl) - (wl * 1e6) * dn_dlambda_per_um

    def amplitude(self, wavelength_m: np.ndarray | float, length_m: float) -> np.ndarray:
        """Complex field transfer coefficient over ``length_m`` of guide."""
        wl = np.asarray(wavelength_m, dtype=float)
        phase = 2.0 * np.pi * self.n_eff(wl) * length_m / wl
        attenuation = db_to_amplitude(self.loss_db_per_cm * length_m * 100.0)
        return attenuation * np.exp(-1j * phase)

    def detuned(self, delta_n_eff: float) -> WaveguideModel:
        """Return a copy with the effective index shifted by ``delta_n_eff``.

        Used by Monte Carlo sampling to emulate width and thickness variation,
        both of which appear to first order as an offset in ``n_eff``.
        """
        c0, c1, c2 = self.n_eff_coeffs
        return replace(self, n_eff_coeffs=(c0 + delta_n_eff, c1, c2))


@dataclass(frozen=True)
class GratingCouplerModel:
    """Fibre-to-chip grating coupler with a Gaussian spectral response.

    Power transmission is::

        T(lam) = 10 ** (-IL/10) * exp(-4 ln2 ((lam - lam_c) / BW) ** 2)

    where ``BW`` is the full width at half maximum. Defaults describe a
    standard uniform SOI coupler: about 3 dB insertion loss at the 1550 nm
    peak and a 40 nm 3 dB bandwidth.
    """

    peak_insertion_loss_db: float = 3.0
    center_m: float = 1.55e-6
    bandwidth_3db_m: float = 40e-9

    def power(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        """Power transmission at the given wavelength(s)."""
        wl = np.asarray(wavelength_m, dtype=float)
        detuning = (wl - self.center_m) / self.bandwidth_3db_m
        envelope = np.exp(-4.0 * np.log(2.0) * detuning**2)
        return db_to_power(self.peak_insertion_loss_db) * envelope

    def amplitude(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        """Field transfer coefficient at the given wavelength(s)."""
        return np.sqrt(self.power(wavelength_m))


@dataclass(frozen=True)
class YBranchModel:
    """Symmetric 1x2 Y-branch.

    Treated as an ideal 50/50 power split with a wavelength-independent
    excess loss. Each output therefore carries a field amplitude of
    ``10 ** (-excess_loss_db / 20) / sqrt(2)``.
    """

    excess_loss_db: float = 0.15

    def amplitude(self, wavelength_m: np.ndarray | float) -> np.ndarray:
        wl = np.asarray(wavelength_m, dtype=float)
        value = db_to_amplitude(self.excess_loss_db) / np.sqrt(2.0)
        return np.full(wl.shape, value, dtype=float) if wl.ndim else np.asarray(value)


@dataclass(frozen=True)
class PDK:
    """A bundle of compact models standing in for a process design kit."""

    waveguide: WaveguideModel = field(default_factory=WaveguideModel)
    grating_coupler: GratingCouplerModel = field(default_factory=GratingCouplerModel)
    y_branch: YBranchModel = field(default_factory=YBranchModel)

    @classmethod
    def default(cls) -> PDK:
        """The nominal SOI strip-waveguide kit used throughout the package."""
        return cls()


@dataclass(frozen=True)
class Component:
    """One component occupying one slot of a circuit.

    ``length_m`` is meaningful only for :attr:`ComponentType.WAVEGUIDE` and is
    ignored otherwise.
    """

    type: ComponentType
    length_m: float = 0.0

    def __post_init__(self) -> None:
        if self.type is ComponentType.WAVEGUIDE and self.length_m <= 0.0:
            raise ValueError(f"waveguide length must be positive, got {self.length_m!r} m")
        if not np.isfinite(self.length_m):
            raise ValueError("component length must be finite")

    @property
    def label(self) -> str:
        if self.type is ComponentType.WAVEGUIDE:
            return f"{self.type.label} ({self.length_m * 1e6:g} um)"
        return self.type.label

    @classmethod
    def waveguide(cls, length_m: float) -> Component:
        return cls(ComponentType.WAVEGUIDE, length_m)

    @classmethod
    def of(cls, type_: ComponentType | str | int, length_m: float = 0.0) -> Component:
        """Build a component from a type, label, or legacy dropdown index."""
        if isinstance(type_, str):
            type_ = ComponentType.from_label(type_)
        return cls(ComponentType(type_), length_m)


def describe(components: Sequence[Component]) -> str:
    """Render a component chain as ``A -> B -> C`` for logs and errors."""
    return " -> ".join(c.label for c in components) if components else "(empty)"
