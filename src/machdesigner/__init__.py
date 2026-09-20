"""MachDesigner -- photonic circuit design and simulation.

A small toolkit for describing, simulating, and plotting silicon photonic
circuits, with a Mach-Zehnder interferometer as the worked example.

Quick start
-----------
>>> from machdesigner import Circuit, SweepSpec, simulate
>>> circuit = Circuit.mzi(long_arm_m=150e-6, short_arm_m=50e-6)
>>> result = simulate(circuit, SweepSpec(points=2000))
>>> round(result.free_spectral_range_m() * 1e9, 2)   # doctest: +SKIP
5.79

The default ``analytic`` backend needs only NumPy. Install the extra
``machdesigner[simphony]`` to run the same circuit through simphony's SiEPIC
compact models instead.
"""

from __future__ import annotations

from .backends import (
    AnalyticBackend,
    BackendUnavailableError,
    SimphonyBackend,
    available_backends,
    get_backend,
    simphony_available,
)
from .components import (
    PDK,
    Component,
    ComponentType,
    GratingCouplerModel,
    WaveguideModel,
    YBranchModel,
)
from .netlist import Circuit, CircuitError, Topology
from .results import MonteCarloResult, SweepResult
from .simulate import UnderSampledSweepWarning, run_monte_carlo, simulate
from .sweep import MonteCarloSpec, SweepSpec

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # circuit description
    "Circuit",
    "CircuitError",
    "Topology",
    "Component",
    "ComponentType",
    # models
    "PDK",
    "WaveguideModel",
    "GratingCouplerModel",
    "YBranchModel",
    # simulation
    "SweepSpec",
    "MonteCarloSpec",
    "simulate",
    "run_monte_carlo",
    "UnderSampledSweepWarning",
    # results
    "SweepResult",
    "MonteCarloResult",
    # backends
    "get_backend",
    "available_backends",
    "AnalyticBackend",
    "SimphonyBackend",
    "BackendUnavailableError",
    "simphony_available",
]
