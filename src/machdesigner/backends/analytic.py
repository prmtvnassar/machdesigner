"""Closed-form transfer-matrix backend.

Every supported topology has an analytic solution, so this backend needs
nothing beyond NumPy. It is the default: it runs in milliseconds, is
deterministic, installs everywhere, and its predictions can be checked against
textbook interferometer relations -- which is exactly what the test suite does.

The physics is documented in ``docs/physics.md``; the short version is that
each component contributes a complex field coefficient and the circuit is the
product (or, at the combiner, the sum) of those coefficients.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..components import PDK, ComponentType
from ..netlist import Circuit, Topology
from ..results import MonteCarloResult, SweepResult
from ..sweep import MonteCarloSpec, SweepSpec

__all__ = ["AnalyticBackend"]


class AnalyticBackend:
    """Solve circuits with closed-form field transfer functions."""

    name = "analytic"

    def __init__(self, pdk: PDK | None = None) -> None:
        self.pdk = pdk or PDK.default()

    # ------------------------------------------------------------ interface

    def sweep(self, circuit: Circuit, spec: SweepSpec) -> SweepResult:
        wavelengths = spec.wavelengths_m
        field = self._output_field(circuit, wavelengths, self.pdk)
        power = spec.laser_power_w * np.abs(field) ** 2
        return SweepResult(
            wavelengths_m=wavelengths,
            power_w=power,
            circuit=circuit,
            spec=spec,
            backend=self.name,
            metadata={
                "model": "closed-form transfer function",
                "n_group_at_1550nm": float(self.pdk.waveguide.n_group(1.55e-6)),
                "predicted_fsr_m": _predicted_fsr(circuit, self.pdk),
            },
        )

    def monte_carlo(
        self, circuit: Circuit, spec: SweepSpec, mc: MonteCarloSpec
    ) -> MonteCarloResult:
        rng = mc.generator()
        nominal = self.sweep(circuit, spec)
        runs: list[SweepResult] = []
        for _ in range(mc.runs):
            perturbed_circuit, perturbed_pdk = _perturb(circuit, self.pdk, mc, rng)
            wavelengths = spec.wavelengths_m
            field = self._output_field(perturbed_circuit, wavelengths, perturbed_pdk)
            runs.append(
                SweepResult(
                    wavelengths_m=wavelengths,
                    power_w=spec.laser_power_w * np.abs(field) ** 2,
                    circuit=perturbed_circuit,
                    spec=spec,
                    backend=self.name,
                    metadata={"monte_carlo": True},
                )
            )
        return MonteCarloResult(
            nominal=nominal,
            runs=tuple(runs),
            sigma_n_eff=mc.sigma_n_eff,
            seed=mc.seed,
        )

    # -------------------------------------------------------------- physics

    @staticmethod
    def _output_field(circuit: Circuit, wavelengths: np.ndarray, pdk: PDK) -> np.ndarray:
        """Complex field at the detector for unit launched amplitude."""
        if circuit.topology is Topology.STRAIGHT:
            return _straight_field(circuit, wavelengths, pdk)
        if circuit.topology is Topology.MZI:
            return _mzi_field(circuit, wavelengths, pdk)
        raise NotImplementedError(f"the analytic backend cannot solve {circuit.topology!r} yet")


def _straight_field(circuit: Circuit, wl: np.ndarray, pdk: PDK) -> np.ndarray:
    """GC -> waveguide -> GC.

    Pure propagation: the grating couplers shape the envelope and the
    waveguide contributes loss and a phase that the power measurement discards.
    """
    (length,) = circuit.arm_lengths_m
    gc = pdk.grating_coupler.amplitude(wl)
    return gc * pdk.waveguide.amplitude(wl, length) * gc


def _mzi_field(circuit: Circuit, wl: np.ndarray, pdk: PDK) -> np.ndarray:
    """GC -> Y-splitter -> two arms -> Y-combiner -> GC.

    The splitter and combiner each contribute ``1/sqrt(2)`` of amplitude per
    branch, so the interference term carries the familiar factor of one half::

        E_out = gc^2 * y^2 * (t1 e^{-j phi1} + t2 e^{-j phi2})

    With lossless, ideal components this reduces to ``cos^2(dphi / 2)`` in
    power, which the test suite checks directly.
    """
    long_arm, short_arm = circuit.arm_lengths_m
    gc = pdk.grating_coupler.amplitude(wl)
    y = pdk.y_branch.amplitude(wl)
    arm_long = pdk.waveguide.amplitude(wl, long_arm)
    arm_short = pdk.waveguide.amplitude(wl, short_arm)
    return gc * y * (arm_long + arm_short) * y * gc


def _predicted_fsr(circuit: Circuit, pdk: PDK) -> float | None:
    """Textbook free spectral range, ``lambda^2 / (n_g * dL)``.

    Reported as metadata so a user can compare the closed-form prediction with
    the value measured from the simulated trace.
    """
    imbalance = circuit.path_imbalance_m
    if imbalance <= 0:
        return None
    lambda0 = pdk.waveguide.lambda0_m
    n_g = float(pdk.waveguide.n_group(lambda0))
    return float(lambda0**2 / (n_g * imbalance))


def _perturb(
    circuit: Circuit, pdk: PDK, mc: MonteCarloSpec, rng: np.random.Generator
) -> tuple[Circuit, PDK]:
    """Draw one process realisation.

    Each waveguide gets an independent effective-index offset. Because a
    :class:`~machdesigner.components.PDK` holds one waveguide model shared by
    every arm, per-arm variation is applied by nudging the arm *lengths* by the
    optical-path amount equivalent to the index offset, which is exactly
    equivalent for the phase term and keeps the model immutable.
    """
    lengths = list(circuit.arm_lengths_m)
    new_lengths: list[float] = []
    n_eff_nominal = float(pdk.waveguide.n_eff(pdk.waveguide.lambda0_m))
    for length in lengths:
        delta_n = rng.normal(0.0, mc.sigma_n_eff) if mc.sigma_n_eff else 0.0
        delta_l = rng.normal(0.0, mc.sigma_length_m) if mc.sigma_length_m else 0.0
        # An index offset dn over length L shifts optical path by dn * L, which
        # is the same phase as a length change of dn * L / n_eff.
        equivalent = length * delta_n / n_eff_nominal
        new_lengths.append(max(length + delta_l + equivalent, 1e-12))

    components = []
    queue = iter(new_lengths)
    for component in circuit.components:
        if component.type is ComponentType.WAVEGUIDE:
            components.append(replace(component, length_m=next(queue)))
        else:
            components.append(component)
    return Circuit(components=tuple(components), topology=circuit.topology), pdk
