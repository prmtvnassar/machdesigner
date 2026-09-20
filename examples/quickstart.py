#!/usr/bin/env python3
"""A tour of the library in one file.

    python examples/quickstart.py

Covers: building and validating circuits, running a sweep, reading the
metrics, process variation, and swapping the simulation backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from machdesigner import (
    Circuit,
    CircuitError,
    Component,
    MonteCarloSpec,
    SweepSpec,
    available_backends,
    run_monte_carlo,
    simulate,
)


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    section("1. Build a circuit")
    circuit = Circuit.mzi(long_arm_m=150e-6, short_arm_m=50e-6)
    print(circuit)
    print(f"arm imbalance: {circuit.path_imbalance_m * 1e6:g} um")

    section("2. Invalid circuits explain themselves")
    try:
        Circuit.from_components([Component.waveguide(50e-6), Component.of("gc-output")])
    except CircuitError as exc:
        print(exc)

    section("3. Sweep")
    spec = SweepSpec(start_m=1500e-9, stop_m=1600e-9, points=4000)
    result = simulate(circuit, spec)
    print(f"insertion loss   : {result.insertion_loss_db:.3f} dB")
    print(f"extinction ratio : {result.extinction_ratio_db:.2f} dB")
    print(f"FSR (measured)   : {result.free_spectral_range_m() * 1e9:.3f} nm")
    print(f"FSR (theory)     : {result.metadata['predicted_fsr_m'] * 1e9:.3f} nm")

    section("4. Free spectral range scales as 1 / imbalance")
    for imbalance_um in (50, 100, 200, 400):
        c = Circuit.mzi(50e-6 + imbalance_um * 1e-6, 50e-6)
        r = simulate(c, SweepSpec(points=200 * imbalance_um))
        print(f"  dL = {imbalance_um:4d} um -> FSR {r.free_spectral_range_m() * 1e9:6.3f} nm")

    section("5. Process variation")
    ensemble = run_monte_carlo(circuit, spec, MonteCarloSpec(runs=50, seed=7))
    stats = ensemble.fsr_statistics_m()
    print(
        f"FSR across 50 runs: {stats['mean_m'] * 1e9:.3f} nm +/- {stats['std_m'] * 1e12:.1f} pm"
    )

    section("6. Backends")
    for name in available_backends():
        r = simulate(circuit, spec, backend=name)
        print(
            f"  {name:9s} IL {r.insertion_loss_db:6.3f} dB   "
            f"FSR {r.free_spectral_range_m() * 1e9:.3f} nm"
        )


if __name__ == "__main__":
    main()
