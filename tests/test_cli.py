"""Command-line interface behaviour, including exit codes and file output."""

from __future__ import annotations

import io
import json

import pytest

from machdesigner.cli import main


def run(args, capsys=None) -> tuple[int, str]:
    stream = io.StringIO()
    code = main(args, stream=stream)
    return code, stream.getvalue()


# ------------------------------------------------------------------ basics


def test_no_command_prints_help():
    code, output = run([])
    assert code == 0
    assert "usage:" in output


def test_backends_lists_the_analytic_engine():
    code, output = run(["backends"])
    assert code == 0
    assert "analytic" in output


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0


# ------------------------------------------------------------------- sweep


def test_sweep_reports_a_summary():
    code, output = run(["sweep", "--points", "2000"])
    assert code == 0
    assert "insertion loss" in output
    assert "free spectral range" in output


def test_sweep_writes_every_requested_artifact(tmp_path):
    png = tmp_path / "plot.png"
    js = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"
    code, _ = run(
        [
            "sweep",
            "--points",
            "500",
            "--plot",
            str(png),
            "--json",
            str(js),
            "--csv",
            str(csv_path),
        ]
    )
    assert code == 0
    # A real figure, not the blank canvas the original tool produced.
    assert png.stat().st_size > 10_000
    assert json.loads(js.read_text())["circuit"]["topology"] == "mzi"
    assert csv_path.read_text().startswith("wavelength_m,")


def test_sweep_creates_missing_output_directories(tmp_path):
    target = tmp_path / "nested" / "deep" / "plot.png"
    code, _ = run(["sweep", "--points", "200", "--plot", str(target)])
    assert code == 0 and target.exists()


def test_straight_topology_runs():
    code, output = run(["sweep", "--topology", "straight", "--short", "250"])
    assert code == 0
    assert "straight waveguide" in output


def test_quiet_suppresses_the_summary(tmp_path):
    code, output = run(["sweep", "--points", "200", "--quiet"])
    assert code == 0
    assert output.strip() == ""


# -------------------------------------------------------------- components


def test_explicit_component_chain_is_accepted():
    code, output = run(
        ["sweep", "--components", "gc-input,waveguide:120,gc-output", "--points", "300"]
    )
    assert code == 0
    assert "straight waveguide" in output


def test_component_chain_overrides_topology_shorthand():
    code, output = run(
        [
            "sweep",
            "--topology",
            "mzi",
            "--components",
            "gc-input,waveguide:120,gc-output",
            "--points",
            "300",
        ]
    )
    assert code == 0
    assert "straight waveguide" in output


def test_waveguide_without_a_length_is_rejected(capsys):
    with pytest.raises(SystemExit) as info:
        main(["sweep", "--components", "gc-input,waveguide,gc-output"])
    assert info.value.code == 2
    assert "needs a length" in capsys.readouterr().err


def test_non_waveguide_with_a_length_is_rejected(capsys):
    with pytest.raises(SystemExit) as info:
        main(["sweep", "--components", "gc-input:5,waveguide:10,gc-output"])
    assert info.value.code == 2
    assert "does not take a length" in capsys.readouterr().err


def test_unknown_component_name_is_rejected(capsys):
    with pytest.raises(SystemExit) as info:
        main(["sweep", "--components", "gc-input,ring-resonator,gc-output"])
    assert info.value.code == 2


# ------------------------------------------------------------ error paths


def test_invalid_circuit_exits_one_and_explains(capsys):
    code = main(["sweep", "--components", "waveguide:50,gc-output"])
    assert code == 1
    assert "must start with an input grating coupler" in capsys.readouterr().err


def test_splitter_without_combiner_exits_one(capsys):
    code = main(
        ["sweep", "--components", "gc-input,y-splitter,waveguide:1,waveguide:2,gc-output"]
    )
    assert code == 1
    assert "Y-combiner" in capsys.readouterr().err


def test_reversed_sweep_bounds_exit_one(capsys):
    code = main(["sweep", "--start", "1600", "--stop", "1500"])
    assert code == 1
    assert "must exceed start" in capsys.readouterr().err


def test_unknown_backend_is_rejected_by_argparse():
    with pytest.raises(SystemExit):
        main(["sweep", "--backend", "nonexistent"])


# ------------------------------------------------------------ monte carlo


def test_monte_carlo_runs_and_reports_spread(tmp_path):
    png = tmp_path / "mc.png"
    code, output = run(
        ["monte-carlo", "--runs", "12", "--seed", "3", "--points", "800", "--plot", str(png)]
    )
    assert code == 0
    assert "12 runs" in output
    assert "FSR across runs" in output
    assert png.stat().st_size > 10_000


def test_monte_carlo_is_reproducible_across_invocations(tmp_path):
    outputs = []
    for index in range(2):
        path = tmp_path / f"mc{index}.json"
        run(
            [
                "monte-carlo",
                "--runs",
                "5",
                "--seed",
                "11",
                "--points",
                "300",
                "--json",
                str(path),
                "--quiet",
            ]
        )
        outputs.append(json.loads(path.read_text())["monte_carlo"]["traces"])
    assert outputs[0] == outputs[1]


# --------------------------------------------------------------- compare


def test_compare_runs_every_installed_backend():
    code, output = run(["compare", "--points", "400"])
    assert code == 0
    assert "[analytic]" in output


# -------------------------------------------------------------- degradation


def test_gui_without_pyqt5_explains_instead_of_traceback(monkeypatch, capsys):
    """Missing PyQt5 must produce an install hint, not an ImportError trace.

    machdesigner.gui imports PyQt5 lazily, so an earlier version of this code
    imported the package successfully and only blew up inside launch().
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PyQt5" or name.startswith("PyQt5."):
            raise ImportError("No module named 'PyQt5'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    code = main(["gui"])
    assert code == 2
    assert "pip install 'machdesigner[gui]'" in capsys.readouterr().err


def test_requesting_simphony_without_it_installed_gives_a_hint(monkeypatch, capsys):
    import machdesigner.backends.simphony_backend as module
    from machdesigner.backends.base import BackendUnavailableError

    def boom():
        raise BackendUnavailableError(module._INSTALL_HINT)

    monkeypatch.setattr(module, "_import_simphony", boom)
    code = main(["sweep", "--backend", "simphony", "--points", "2000"])
    assert code == 2
    assert "pip install" in capsys.readouterr().err
