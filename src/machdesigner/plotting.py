"""Figure builders for sweep and Monte Carlo results.

Every function here takes an optional ``ax`` and returns a Figure. Nothing
calls ``plt.show()`` or ``plt.savefig()``: the caller decides whether a figure
goes to a file, into a Qt canvas, or nowhere.

That separation is deliberate. The original tool called ``plt.show()`` and then
``plt.savefig()`` on the same figure. Interactively, ``show()`` blocks until the
window is closed and leaves no current figure behind, so the subsequent
``savefig`` wrote a blank canvas -- which is what the GUI then displayed. The
same call also blocked the Qt event loop for as long as the window was open.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .results import MonteCarloResult, SweepResult

__all__ = [
    "plot_comparison",
    "plot_monte_carlo",
    "plot_sweep",
]

_DEFAULT_FIGSIZE = (7.0, 4.2)


def _axes(ax: Axes | None, figsize: tuple[float, float] = _DEFAULT_FIGSIZE) -> Axes:
    """Return an axes to draw on, creating a detached figure if needed."""
    if ax is not None:
        return ax
    # Figure() rather than plt.figure() so we never touch pyplot's global
    # state; that global state is what made the original plots accumulate
    # lines from previous runs.
    figure = Figure(figsize=figsize, layout="constrained")
    return figure.add_subplot(111)


def _decorate(ax: Axes, title: str, *, db: bool) -> None:
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (dB)" if db else "Transmission")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linewidth=0.6)


def plot_sweep(
    result: SweepResult,
    ax: Axes | None = None,
    *,
    db: bool = False,
    title: str | None = None,
    label: str | None = None,
    annotate: bool = True,
) -> Figure:
    """Plot a single transmission spectrum.

    Parameters
    ----------
    db:
        Plot in decibels rather than linear transmission.
    annotate:
        Add a corner box with insertion loss, extinction ratio, and the
        measured free spectral range.
    """
    ax = _axes(ax)
    wl_nm = result.wavelengths_m * 1e9
    y = result.transmission_db if db else result.transmission
    ax.plot(wl_nm, y, linewidth=1.4, label=label or result.circuit.topology.description)

    _decorate(
        ax,
        title or f"{result.circuit.topology.description} ({result.backend})",
        db=db,
    )
    if not db:
        ax.set_ylim(bottom=0.0)

    if annotate:
        ax.text(
            0.02,
            0.02,
            _summary_text(result),
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.75, "lw": 0.5},
        )
    if label:
        ax.legend(fontsize=8)
    return ax.figure


def _summary_text(result: SweepResult) -> str:
    lines = [
        f"IL {result.insertion_loss_db:.2f} dB",
        f"ER {result.extinction_ratio_db:.1f} dB",
    ]
    fsr = result.free_spectral_range_m()
    if fsr is not None:
        lines.append(f"FSR {fsr * 1e9:.2f} nm")
    predicted = result.metadata.get("predicted_fsr_m")
    if predicted:
        lines.append(f"FSR (theory) {predicted * 1e9:.2f} nm")
    return "\n".join(lines)


def plot_monte_carlo(
    result: MonteCarloResult,
    ax: Axes | None = None,
    *,
    db: bool = False,
    title: str | None = None,
    max_traces: int = 25,
    show_band: bool = True,
) -> Figure:
    """Plot a Monte Carlo ensemble against its nominal trace.

    Individual runs are drawn faintly, capped at ``max_traces`` so a large
    ensemble stays readable, with a 5th-95th percentile band and the nominal
    curve on top.
    """
    ax = _axes(ax)
    wl_nm = result.nominal.wavelengths_m * 1e9

    if show_band and len(result.runs) > 2:
        low, high = result.percentile_band()
        if db:
            floor = np.finfo(float).tiny
            low = 10 * np.log10(np.maximum(low, floor))
            high = 10 * np.log10(np.maximum(high, floor))
        ax.fill_between(wl_nm, low, high, alpha=0.2, linewidth=0, label="5-95th percentile")

    for index, run in enumerate(result.runs[:max_traces]):
        y = run.transmission_db if db else run.transmission
        ax.plot(
            wl_nm,
            y,
            linewidth=0.7,
            alpha=0.45,
            color="tab:blue",
            label="Monte Carlo runs" if index == 0 else None,
        )

    nominal_y = result.nominal.transmission_db if db else result.nominal.transmission
    ax.plot(wl_nm, nominal_y, color="black", linewidth=1.6, label="Nominal")

    hidden = max(0, len(result.runs) - max_traces)
    suffix = f" ({hidden} more not drawn)" if hidden else ""
    _decorate(
        ax,
        title or f"Monte Carlo, {len(result.runs)} runs{suffix}",
        db=db,
    )
    if not db:
        ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8, loc="upper right")
    return ax.figure


def plot_comparison(
    results: Iterable[SweepResult],
    ax: Axes | None = None,
    *,
    db: bool = False,
    title: str = "Backend comparison",
) -> Figure:
    """Overlay several sweeps, one line each.

    Used to check the analytic and simphony backends against each other.
    """
    ax = _axes(ax)
    for result in results:
        ax.plot(
            result.wavelengths_m * 1e9,
            result.transmission_db if db else result.transmission,
            linewidth=1.3,
            label=result.backend,
        )
    _decorate(ax, title, db=db)
    if not db:
        ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=8)
    return ax.figure
