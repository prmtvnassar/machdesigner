"""Backend delegating to simphony's SiEPIC compact models.

This wraps the engine the original MachDesigner used. It is optional: install
it with ``pip install machdesigner[simphony]``.

Version note
------------
simphony 0.7 replaced the ``simphony.simulation`` API (``Laser``, ``Detector``,
``Simulation``) with a sax-based one. This adapter targets the 0.6.x API that
the original tool was written against, and says so explicitly rather than
failing with a confusing ``ImportError`` if a newer simphony is installed.
"""

from __future__ import annotations

import numpy as np

from ..netlist import Circuit, Topology
from ..results import MonteCarloResult, SweepResult
from ..sweep import SPEED_OF_LIGHT, MonteCarloSpec, SweepSpec
from .base import BackendUnavailableError

__all__ = ["SimphonyBackend", "simphony_available"]

_INSTALL_HINT = (
    "The simphony backend needs simphony 0.6.x:\n"
    "    pip install 'simphony>=0.6,<0.7'\n"
    "simphony 0.7 removed the simulation API this adapter uses. "
    "The default 'analytic' backend has no extra dependencies."
)


def _import_simphony():
    """Import the 0.6.x simphony API or explain why we cannot."""
    try:
        from simphony.libraries import siepic
        from simphony.simulation import (
            Detector,
            Laser,
            Simulation,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BackendUnavailableError(f"{exc}\n\n{_INSTALL_HINT}") from exc
    return siepic, Simulation, Laser, Detector


def simphony_available() -> bool:
    """Whether the simphony backend can run in this environment."""
    try:
        _import_simphony()
    except BackendUnavailableError:
        return False
    return True


class SimphonyBackend:
    """Solve circuits by building and sampling a simphony netlist."""

    name = "simphony"

    def __init__(self) -> None:
        # Fail at construction rather than halfway through a long sweep.
        self._siepic, self._Simulation, self._Laser, self._Detector = _import_simphony()

    # ------------------------------------------------------------ interface

    def sweep(self, circuit: Circuit, spec: SweepSpec) -> SweepResult:
        freqs, power = self._run(circuit, spec, monte_carlo=False)
        return self._to_result(circuit, spec, freqs, power, {"samples": 1})

    def monte_carlo(
        self, circuit: Circuit, spec: SweepSpec, mc: MonteCarloSpec
    ) -> MonteCarloResult:
        nominal = self.sweep(circuit, spec)
        runs: list[SweepResult] = []
        for _ in range(mc.runs):
            freqs, power = self._run(circuit, spec, monte_carlo=True)
            runs.append(self._to_result(circuit, spec, freqs, power, {"monte_carlo": True}))
        return MonteCarloResult(
            nominal=nominal,
            runs=tuple(runs),
            sigma_n_eff=float("nan"),  # simphony owns its own variation model
            seed=None,
        )

    # --------------------------------------------------------------- engine

    def _build(self, circuit: Circuit):
        """Instantiate and wire simphony components for ``circuit``."""
        siepic = self._siepic
        gc_input = siepic.GratingCoupler()
        gc_output = siepic.GratingCoupler()

        if circuit.topology is Topology.STRAIGHT:
            (length,) = circuit.arm_lengths_m
            waveguide = siepic.Waveguide(length=length)
            waveguide.multiconnect(gc_input, gc_output)
            return gc_input, gc_output

        if circuit.topology is Topology.MZI:
            long_m, short_m = circuit.arm_lengths_m
            splitter = siepic.YBranch()
            combiner = siepic.YBranch()
            arm_long = siepic.Waveguide(length=long_m)
            arm_short = siepic.Waveguide(length=short_m)
            splitter.multiconnect(gc_input, arm_long, arm_short)
            combiner.multiconnect(gc_output, arm_short, arm_long)
            return gc_input, gc_output

        raise NotImplementedError(f"the simphony backend cannot build {circuit.topology!r}")

    def _run(
        self, circuit: Circuit, spec: SweepSpec, *, monte_carlo: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        gc_input, gc_output = self._build(circuit)

        if monte_carlo:
            for component in gc_input.circuit:
                component.regenerate_monte_carlo_parameters()

        with self._Simulation() as sim:
            laser = self._Laser(power=spec.laser_power_w)
            laser.wlsweep(spec.start_m, spec.stop_m)
            laser.connect(gc_input)
            self._Detector().connect(gc_output)
            if monte_carlo:
                sim.monte_carlo(True)
            # NOTE: sample() must be called inside the context manager. The
            # original source_code.py called it outside, which raised
            # AttributeError on a detached circuit.
            data = sim.sample()
            freqs = np.asarray(sim.freqs, dtype=float)

        return freqs, np.asarray(data, dtype=float)[:, 0, 0]

    def _to_result(
        self,
        circuit: Circuit,
        spec: SweepSpec,
        freqs: np.ndarray,
        power: np.ndarray,
        metadata: dict,
    ) -> SweepResult:
        """Convert simphony's frequency-ordered output to a wavelength sweep.

        simphony returns ascending frequency, which is descending wavelength;
        we flip so every backend hands back ascending wavelength.
        """
        wavelengths = SPEED_OF_LIGHT / freqs
        order = np.argsort(wavelengths)
        return SweepResult(
            wavelengths_m=wavelengths[order],
            power_w=power[order],
            circuit=circuit,
            spec=spec,
            backend=self.name,
            metadata={
                "engine": "simphony/siepic",
                # simphony picks its own sweep resolution, so SweepSpec.points
                # is advisory here; record what we actually got.
                "points_from_engine": int(wavelengths.size),
                "requested_points": spec.points,
                **metadata,
            },
        )
