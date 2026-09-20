"""Command-line interface.

Makes the simulator scriptable and testable without a display, which is what
lets continuous integration exercise the physics on every commit.

Units follow the conventions an engineer would type: lengths in micrometres,
wavelengths in nanometres, power in milliwatts. The library itself is
strictly SI; conversion happens here at the boundary.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .backends import BACKEND_NAMES, BackendUnavailableError, available_backends
from .components import Component, ComponentType
from .netlist import Circuit, CircuitError
from .simulate import UnderSampledSweepWarning, run_monte_carlo, simulate
from .sweep import MonteCarloSpec, SweepSpec

__all__ = ["build_parser", "main"]

_EPILOG = """\
examples:
  # default unbalanced MZI, save a plot and the raw trace
  machdesigner sweep --long 150 --short 50 --plot mzi.png --csv mzi.csv

  # an arbitrary component chain
  machdesigner sweep --components gc-input,y-splitter,waveguide:150,\\
waveguide:50,y-combiner,gc-output

  # 200-run yield study with a fixed seed
  machdesigner monte-carlo --long 150 --short 50 --runs 200 --seed 7 --plot mc.png

  # cross-check the two engines
  machdesigner compare --long 150 --short 50 --plot compare.png
"""


# --------------------------------------------------------------- arg parsing


def _parse_components(text: str) -> list[Component]:
    """Parse ``gc-input,waveguide:150,gc-output`` into components.

    A waveguide carries its length in micrometres after a colon.
    """
    components: list[Component] = []
    for index, token in enumerate(text.split(","), start=1):
        token = token.strip()
        if not token:
            continue
        name, _, length = token.partition(":")
        try:
            type_ = ComponentType.from_label(name)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"component {index}: {exc}") from None
        if type_ is ComponentType.WAVEGUIDE:
            if not length:
                raise argparse.ArgumentTypeError(
                    f"component {index} ({name}) needs a length, "
                    f"e.g. 'waveguide:150' for 150 um"
                )
            try:
                length_um = float(length)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"component {index}: {length!r} is not a number of micrometres"
                ) from None
            components.append(Component.waveguide(length_um * 1e-6))
        else:
            if length:
                raise argparse.ArgumentTypeError(
                    f"component {index} ({name}) does not take a length"
                )
            components.append(Component.of(type_))
    return components


def _add_circuit_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("circuit")
    group.add_argument(
        "--components",
        metavar="CHAIN",
        help="explicit component chain, comma separated, waveguide lengths in "
        "um after a colon (overrides --topology/--long/--short)",
    )
    group.add_argument(
        "--topology",
        choices=("mzi", "straight"),
        default="mzi",
        help="shorthand circuit to build (default: mzi)",
    )
    group.add_argument(
        "--long",
        type=float,
        default=150.0,
        metavar="UM",
        help="long arm length in micrometres (default: 150)",
    )
    group.add_argument(
        "--short",
        type=float,
        default=50.0,
        metavar="UM",
        help="short arm length, or the guide length for --topology straight (default: 50)",
    )


def _add_sweep_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("sweep")
    group.add_argument(
        "--start",
        type=float,
        default=1500.0,
        metavar="NM",
        help="sweep start wavelength in nm (default: 1500)",
    )
    group.add_argument(
        "--stop",
        type=float,
        default=1600.0,
        metavar="NM",
        help="sweep stop wavelength in nm (default: 1600)",
    )
    group.add_argument(
        "--points",
        type=int,
        default=2000,
        metavar="N",
        help="number of sweep samples (default: 2000)",
    )
    group.add_argument(
        "--power",
        type=float,
        default=20.0,
        metavar="MW",
        help="laser power in milliwatts (default: 20)",
    )
    group.add_argument(
        "--backend",
        choices=BACKEND_NAMES,
        default="auto",
        help="simulation engine (default: auto, which selects analytic)",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")
    group.add_argument("--plot", metavar="PATH", help="write a PNG/PDF/SVG figure")
    group.add_argument("--json", metavar="PATH", help="write results as JSON")
    group.add_argument("--csv", metavar="PATH", help="write the trace as CSV")
    group.add_argument("--db", action="store_true", help="plot in decibels")
    group.add_argument("--dpi", type=int, default=150, help="figure resolution (default: 150)")
    group.add_argument("-q", "--quiet", action="store_true", help="suppress the summary")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="machdesigner",
        description="Simulate silicon photonic circuits from the command line.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"machdesigner {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    sweep_parser = subparsers.add_parser(
        "sweep",
        help="run a single wavelength sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_circuit_arguments(sweep_parser)
    _add_sweep_arguments(sweep_parser)
    _add_output_arguments(sweep_parser)

    mc_parser = subparsers.add_parser(
        "monte-carlo",
        help="run a process-variation study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_circuit_arguments(mc_parser)
    _add_sweep_arguments(mc_parser)
    _add_output_arguments(mc_parser)
    mc_group = mc_parser.add_argument_group("monte carlo")
    mc_group.add_argument("--runs", type=int, default=20, help="number of samples")
    mc_group.add_argument(
        "--sigma-neff",
        type=float,
        default=2e-3,
        help="standard deviation of the per-waveguide effective-index perturbation",
    )
    mc_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed; fixed by default so runs are reproducible",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="run the same circuit on every available backend",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_circuit_arguments(compare_parser)
    _add_sweep_arguments(compare_parser)
    _add_output_arguments(compare_parser)

    subparsers.add_parser("backends", help="list available simulation engines")
    subparsers.add_parser("gui", help="launch the graphical designer")

    return parser


# ------------------------------------------------------------------ helpers


def _circuit_from(args: argparse.Namespace) -> Circuit:
    if args.components:
        return Circuit.from_components(_parse_components(args.components))
    if args.topology == "straight":
        return Circuit.straight(args.short * 1e-6)
    return Circuit.mzi(long_arm_m=args.long * 1e-6, short_arm_m=args.short * 1e-6)


def _spec_from(args: argparse.Namespace) -> SweepSpec:
    return SweepSpec(
        start_m=args.start * 1e-9,
        stop_m=args.stop * 1e-9,
        points=args.points,
        laser_power_w=args.power * 1e-3,
    )


def _save_figure(figure, path: str, dpi: int) -> Path:
    destination = Path(path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi)
    return destination


def _summarise(result, stream) -> None:
    print(f"  circuit        : {result.circuit}", file=stream)
    print(f"  backend        : {result.backend}", file=stream)
    print(f"  insertion loss : {result.insertion_loss_db:.3f} dB", file=stream)
    print(f"  extinction     : {result.extinction_ratio_db:.2f} dB", file=stream)
    fsr = result.free_spectral_range_m()
    if fsr is None:
        print("  free spectral range: not resolved (balanced or non-interfering)", file=stream)
    else:
        print(f"  free spectral range: {fsr * 1e9:.3f} nm", file=stream)
        predicted = result.metadata.get("predicted_fsr_m")
        if predicted:
            error = abs(fsr - predicted) / predicted * 100
            print(
                f"  theory             : {predicted * 1e9:.3f} nm ({error:.2f}% from measured)",
                file=stream,
            )


# ----------------------------------------------------------------- commands


def _cmd_sweep(args: argparse.Namespace, stream) -> int:
    from .plotting import plot_sweep  # imported lazily so --help stays fast

    circuit = _circuit_from(args)
    result = simulate(circuit, _spec_from(args), backend=args.backend)

    if not args.quiet:
        print("sweep complete", file=stream)
        _summarise(result, stream)

    if args.plot:
        figure = plot_sweep(result, db=args.db)
        path = _save_figure(figure, args.plot, args.dpi)
        if not args.quiet:
            print(f"  figure  -> {path}", file=stream)
    if args.json:
        path = result.write_json(args.json)
        if not args.quiet:
            print(f"  json    -> {path}", file=stream)
    if args.csv:
        path = result.write_csv(args.csv)
        if not args.quiet:
            print(f"  csv     -> {path}", file=stream)
    return 0


def _cmd_monte_carlo(args: argparse.Namespace, stream) -> int:
    from .plotting import plot_monte_carlo

    circuit = _circuit_from(args)
    result = run_monte_carlo(
        circuit,
        _spec_from(args),
        MonteCarloSpec(runs=args.runs, sigma_n_eff=args.sigma_neff, seed=args.seed),
        backend=args.backend,
    )

    if not args.quiet:
        print(f"monte carlo complete: {len(result.runs)} runs, seed {result.seed}", file=stream)
        _summarise(result.nominal, stream)
        stats = result.fsr_statistics_m()
        if stats:
            print(
                f"  FSR across runs: {stats['mean_m'] * 1e9:.3f} nm "
                f"+/- {stats['std_m'] * 1e12:.1f} pm "
                f"(n={int(stats['samples'])})",
                file=stream,
            )

    if args.plot:
        figure = plot_monte_carlo(result, db=args.db)
        path = _save_figure(figure, args.plot, args.dpi)
        if not args.quiet:
            print(f"  figure  -> {path}", file=stream)
    if args.json:
        path = result.write_json(args.json)
        if not args.quiet:
            print(f"  json    -> {path}", file=stream)
    if args.csv:
        path = result.nominal.write_csv(args.csv)
        if not args.quiet:
            print(f"  csv     -> {path} (nominal trace)", file=stream)
    return 0


def _cmd_compare(args: argparse.Namespace, stream) -> int:
    from .plotting import plot_comparison

    circuit = _circuit_from(args)
    spec = _spec_from(args)
    results = []
    for name in available_backends():
        result = simulate(circuit, spec, backend=name)
        results.append(result)
        if not args.quiet:
            print(f"[{name}]", file=stream)
            _summarise(result, stream)
            print(file=stream)

    if len(results) < 2 and not args.quiet:
        print(
            "only one backend is installed; install machdesigner[simphony] to compare engines",
            file=stream,
        )

    if args.plot:
        figure = plot_comparison(results, db=args.db)
        path = _save_figure(figure, args.plot, args.dpi)
        if not args.quiet:
            print(f"figure -> {path}", file=stream)
    return 0


def _cmd_backends(stream) -> int:
    installed = available_backends()
    print("available backends:", file=stream)
    for name in ("analytic", "simphony"):
        mark = "yes" if name in installed else "no "
        note = "" if name in installed else "   (pip install 'machdesigner[simphony]')"
        print(f"  [{mark}] {name}{note}", file=stream)
    print(f"\ndefault: {installed[0]}", file=stream)
    return 0


def _cmd_gui(stream) -> int:
    # machdesigner.gui imports PyQt5 lazily, so importing it proves nothing.
    # Probe the dependency itself; that way a genuine ImportError from inside
    # the GUI still propagates as a bug instead of being reported as a
    # missing install.
    try:
        import PyQt5  # noqa: F401
    except ImportError as exc:
        print(
            f"the GUI needs PyQt5:\n    pip install 'machdesigner[gui]'\n({exc})",
            file=sys.stderr,
        )
        return 2

    from .gui import launch

    return launch()


# -------------------------------------------------------------------- entry


def main(argv: Sequence[str] | None = None, stream=None) -> int:
    """Run the CLI. Returns a process exit code."""
    stream = stream or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(stream)
        return 0
    if args.command == "backends":
        return _cmd_backends(stream)
    if args.command == "gui":
        return _cmd_gui(stream)

    # Surface under-sampling as a visible warning rather than letting it pass
    # silently into a misleading plot.
    warnings.simplefilter("always", UnderSampledSweepWarning)

    try:
        if args.command == "sweep":
            return _cmd_sweep(args, stream)
        if args.command == "monte-carlo":
            return _cmd_monte_carlo(args, stream)
        if args.command == "compare":
            return _cmd_compare(args, stream)
    except argparse.ArgumentTypeError as exc:
        # --components is parsed after argparse has finished, so a malformed
        # value has to be routed back through the parser to get the usual
        # usage message and exit status rather than a traceback.
        parser.error(str(exc))
    except CircuitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except BackendUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.print_help(stream)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
