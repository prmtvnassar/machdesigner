#!/usr/bin/env python3
"""Regenerate the figures used in the README.

    python examples/generate_figures.py

Writes into ``docs/images/``. Run it after changing any default model
parameter so the documentation keeps matching the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from machdesigner import (
    Circuit,
    MonteCarloSpec,
    SweepSpec,
    available_backends,
    run_monte_carlo,
    simulate,
)
from machdesigner.plotting import (
    plot_comparison,
    plot_monte_carlo,
    plot_sweep,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "images"
DPI = 140


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    circuit = Circuit.mzi(long_arm_m=150e-6, short_arm_m=50e-6)
    spec = SweepSpec(start_m=1500e-9, stop_m=1600e-9, points=4000)

    print("spectrum...")
    result = simulate(circuit, spec)
    plot_sweep(result, title="Unbalanced MZI, 100 um arm imbalance").savefig(
        OUTPUT_DIR / "mzi_spectrum.png", dpi=DPI
    )
    print(
        f"  FSR measured {result.free_spectral_range_m() * 1e9:.3f} nm, "
        f"theory {result.metadata['predicted_fsr_m'] * 1e9:.3f} nm"
    )

    print("monte carlo...")
    ensemble = run_monte_carlo(circuit, spec, MonteCarloSpec(runs=60, sigma_n_eff=2e-3, seed=7))
    # Only a subset of traces is drawn: at realistic variation each run shifts
    # by a sizeable fraction of a fringe, so 60 overlaid curves fill the
    # envelope solid and nothing is legible.
    plot_monte_carlo(
        ensemble,
        title="Process variation, 60 runs (sigma n_eff = 2e-3)",
        max_traces=12,
    ).savefig(OUTPUT_DIR / "monte_carlo.png", dpi=DPI)
    stats = ensemble.fsr_statistics_m()
    print(
        f"  FSR {stats['mean_m'] * 1e9:.3f} nm +/- {stats['std_m'] * 1e12:.1f} pm "
        f"across {int(stats['samples'])} runs"
    )

    print("backend comparison...")
    backends = available_backends()
    results = [simulate(circuit, spec, backend=name) for name in backends]
    plot_comparison(results, title="Analytic vs simphony, identical circuit").savefig(
        OUTPUT_DIR / "backend_comparison.png", dpi=DPI
    )
    for name, item in zip(backends, results, strict=True):
        fsr = item.free_spectral_range_m()
        print(f"  {name:9s} FSR {fsr * 1e9:.3f} nm" if fsr else f"  {name}: no fringes")
    if len(results) < 2:
        print("  (only one backend installed; install machdesigner[simphony])")

    print(f"\nwrote figures to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
