"""GUI tests, run against Qt's offscreen platform.

These skip when PyQt5 is unavailable. They cover the behaviour that the
original tool got wrong: live validation, per-slot length fields, figures with
actual content, and simulation happening off the UI thread.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is not installed")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEventLoop, QTimer
from PyQt5.QtWidgets import QApplication

from machdesigner.components import ComponentType


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def window(qapp):
    from machdesigner.gui.main_window import MainWindow

    win = MainWindow()
    yield win
    win.close()


def _set_slot(window, index: int, component_type: ComponentType) -> None:
    combo = window.slot_boxes[index]
    combo.setCurrentIndex(combo.findData(component_type))


def _run_and_wait(window, timeout_ms: int = 120_000) -> None:
    """Press Run and pump the event loop until the worker thread finishes."""
    loop = QEventLoop()
    window._on_run()
    assert window._worker is not None, "Run did not start a worker"
    window._worker.finished.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec_()
    window._worker.wait(5000)


# ------------------------------------------------------------ construction


def test_window_opens_with_a_valid_default_circuit(window):
    assert "MachDesigner" in window.windowTitle()
    assert window.run_button.isEnabled()
    assert "Valid circuit" in window.validation_label.text()


def test_all_six_slots_are_populated_by_default(window):
    """The original ignored slot six entirely; the default now fills it."""
    types = [combo.currentData() for combo in window.slot_boxes]
    assert types == [
        ComponentType.GC_INPUT,
        ComponentType.Y_SPLITTER,
        ComponentType.WAVEGUIDE,
        ComponentType.WAVEGUIDE,
        ComponentType.Y_COMBINER,
        ComponentType.GC_OUTPUT,
    ]


def test_length_fields_enable_only_for_waveguide_slots(window):
    enabled = [box.isEnabled() for box in window.length_boxes]
    assert enabled == [False, False, True, True, False, False]


def test_changing_a_slot_to_waveguide_enables_its_length_field(window):
    assert not window.length_boxes[0].isEnabled()
    _set_slot(window, 0, ComponentType.WAVEGUIDE)
    assert window.length_boxes[0].isEnabled()


# -------------------------------------------------------------- validation


def test_invalid_circuit_disables_run_and_explains_why(window):
    _set_slot(window, 0, ComponentType.WAVEGUIDE)
    assert not window.run_button.isEnabled()
    text = window.validation_label.text()
    assert "Cannot simulate" in text
    assert "must start with an input grating coupler" in text


def test_validation_recovers_when_the_circuit_is_fixed(window):
    _set_slot(window, 0, ComponentType.WAVEGUIDE)
    assert not window.run_button.isEnabled()
    _set_slot(window, 0, ComponentType.GC_INPUT)
    assert window.run_button.isEnabled()
    assert "Valid circuit" in window.validation_label.text()


def test_validation_reports_the_arm_imbalance(window):
    window.length_boxes[2].setValue(250.0)
    window.length_boxes[3].setValue(50.0)
    assert "200 um" in window.validation_label.text()


@pytest.mark.parametrize("preset", ["Unbalanced MZI", "Balanced MZI", "Straight waveguide"])
def test_every_preset_is_valid(window, preset):
    window._apply_preset(preset)
    assert window.run_button.isEnabled(), f"preset {preset!r} did not validate"


def test_straight_preset_clears_the_trailing_slots(window):
    window._apply_preset("Straight waveguide")
    types = [combo.currentData() for combo in window.slot_boxes]
    assert types[3:] == [ComponentType.NONE] * 3


# ---------------------------------------------------------------- running


def test_run_produces_results_and_a_non_blank_figure(window, tmp_path):
    window.points_box.setValue(600)
    window.mc_enabled.setChecked(False)
    _run_and_wait(window)

    assert window._last_sweep is not None
    assert window.export_figure_button.isEnabled()

    path = tmp_path / "spectrum.png"
    window.sweep_canvas.save(str(path))
    # The original wrote a blank canvas here because savefig followed show().
    assert path.stat().st_size > 10_000


def test_run_completes_a_monte_carlo_study(window):
    window.points_box.setValue(400)
    window.mc_enabled.setChecked(True)
    window.mc_runs.setValue(4)
    _run_and_wait(window)
    assert window._last_mc is not None
    assert len(window._last_mc.runs) == 4


def test_status_bar_shows_the_summary_after_running(window):
    window.points_box.setValue(400)
    window.mc_enabled.setChecked(False)
    _run_and_wait(window)
    message = window.statusBar().currentMessage()
    assert "IL" in message and "FSR" in message


def test_run_button_is_restored_after_completion(window):
    window.points_box.setValue(300)
    window.mc_enabled.setChecked(False)
    _run_and_wait(window)
    assert window.run_button.isEnabled()
    assert window.run_button.text() == "Run"


def test_repeated_runs_do_not_accumulate_curves(window):
    """The original never cleared the axes, so runs stacked on one another."""
    window.points_box.setValue(300)
    window.mc_enabled.setChecked(False)
    _run_and_wait(window)
    first = len(window.sweep_canvas.axes.lines)
    _run_and_wait(window)
    assert len(window.sweep_canvas.axes.lines) == first


def test_db_toggle_redraws_without_error(window):
    window.points_box.setValue(300)
    window.mc_enabled.setChecked(False)
    _run_and_wait(window)
    window.db_toggle.setChecked(True)
    assert window.sweep_canvas.axes.get_ylabel() == "Transmission (dB)"


def test_export_writes_a_csv(window, tmp_path):
    window.points_box.setValue(250)
    window.mc_enabled.setChecked(False)
    _run_and_wait(window)
    path = tmp_path / "sweep.csv"
    window._last_sweep.write_csv(path)
    assert path.read_text().startswith("wavelength_m,")
