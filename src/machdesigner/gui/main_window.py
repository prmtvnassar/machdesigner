"""The MachDesigner main window.

Rebuilt from the Qt Designer output of the original tool. The substantive
changes:

* Real layouts instead of absolute ``setGeometry`` coordinates, so the window
  resizes and adapts to different font sizes and platforms.
* Each of the six slots owns its own length field, so the sixth dropdown is
  meaningful. The original read arm lengths from two fixed spin boxes and
  never inspected slot six at all.
* Circuits are validated structurally and continuously, so the user sees what
  is wrong while editing rather than a generic message after pressing Run.
* Plots are drawn into embedded canvases; simulations run on a worker thread.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..backends import available_backends
from ..components import Component, ComponentType
from ..netlist import Circuit, CircuitError
from ..plotting import plot_monte_carlo, plot_sweep
from ..results import MonteCarloResult, SweepResult
from ..sweep import MonteCarloSpec, SweepSpec
from .canvas import PlotCanvas
from .worker import SimulationWorker

__all__ = ["MainWindow"]

SLOT_COUNT = 6

#: Starting configuration: the canonical unbalanced MZI, filling all six slots.
_DEFAULT_SLOTS: list[tuple[ComponentType, float]] = [
    (ComponentType.GC_INPUT, 0.0),
    (ComponentType.Y_SPLITTER, 0.0),
    (ComponentType.WAVEGUIDE, 150.0),
    (ComponentType.WAVEGUIDE, 50.0),
    (ComponentType.Y_COMBINER, 0.0),
    (ComponentType.GC_OUTPUT, 0.0),
]

_PRESETS: dict[str, list[tuple[ComponentType, float]]] = {
    "Unbalanced MZI": _DEFAULT_SLOTS,
    "Balanced MZI": [
        (ComponentType.GC_INPUT, 0.0),
        (ComponentType.Y_SPLITTER, 0.0),
        (ComponentType.WAVEGUIDE, 100.0),
        (ComponentType.WAVEGUIDE, 100.0),
        (ComponentType.Y_COMBINER, 0.0),
        (ComponentType.GC_OUTPUT, 0.0),
    ],
    "Straight waveguide": [
        (ComponentType.GC_INPUT, 0.0),
        (ComponentType.WAVEGUIDE, 100.0),
        (ComponentType.GC_OUTPUT, 0.0),
        (ComponentType.NONE, 0.0),
        (ComponentType.NONE, 0.0),
        (ComponentType.NONE, 0.0),
    ],
}


class MainWindow(QMainWindow):
    """Top-level designer window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("MachDesigner - Photonic Circuit Designer")
        self.resize(1020, 780)

        self._worker: SimulationWorker | None = None
        self._last_sweep: SweepResult | None = None
        self._last_mc: MonteCarloResult | None = None

        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setSpacing(10)

        root.addWidget(self._build_header())
        root.addWidget(self._build_circuit_group())

        middle = QHBoxLayout()
        middle.addWidget(self._build_sweep_group(), 3)
        middle.addWidget(self._build_monte_carlo_group(), 2)
        root.addLayout(middle)

        root.addWidget(self._build_status_bar_widget())
        root.addLayout(self._build_actions())
        root.addWidget(self._build_plots(), 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

        self._apply_preset("Unbalanced MZI")

    # ------------------------------------------------------------- building

    def _build_header(self) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Photonic Circuit Designer")
        font = QFont()
        font.setPointSize(13)
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        layout.addStretch(1)

        layout.addWidget(QLabel("Preset:"))
        self.preset_box = QComboBox()
        self.preset_box.addItems(list(_PRESETS))
        self.preset_box.setToolTip("Load a known-good starting circuit")
        self.preset_box.activated.connect(
            lambda: self._apply_preset(self.preset_box.currentText())
        )
        layout.addWidget(self.preset_box)
        return container

    def _build_circuit_group(self) -> QGroupBox:
        group = QGroupBox("Circuit")
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(8)

        self.slot_boxes: list[QComboBox] = []
        self.length_boxes: list[QDoubleSpinBox] = []

        for slot in range(SLOT_COUNT):
            label = QLabel(f"Slot {slot + 1}")
            label.setAlignment(Qt.AlignCenter)
            grid.addWidget(label, 0, slot)

            combo = QComboBox()
            for component_type in ComponentType:
                combo.addItem(component_type.label, component_type)
            combo.currentIndexChanged.connect(self._on_circuit_changed)
            grid.addWidget(combo, 1, slot)
            self.slot_boxes.append(combo)

            length = QDoubleSpinBox()
            length.setRange(0.1, 1_000_000.0)
            length.setDecimals(2)
            length.setSingleStep(10.0)
            length.setSuffix(" um")
            length.setValue(100.0)
            length.setToolTip("Waveguide length in micrometres")
            length.valueChanged.connect(self._on_circuit_changed)
            grid.addWidget(length, 2, slot)
            self.length_boxes.append(length)

        return group

    def _build_sweep_group(self) -> QGroupBox:
        group = QGroupBox("Sweep")
        grid = QGridLayout(group)

        self.start_box = QDoubleSpinBox()
        self.start_box.setRange(200.0, 20_000.0)
        self.start_box.setDecimals(1)
        self.start_box.setSuffix(" nm")
        self.start_box.setValue(1500.0)

        self.stop_box = QDoubleSpinBox()
        self.stop_box.setRange(200.0, 20_000.0)
        self.stop_box.setDecimals(1)
        self.stop_box.setSuffix(" nm")
        self.stop_box.setValue(1600.0)

        self.points_box = QSpinBox()
        self.points_box.setRange(2, 200_000)
        self.points_box.setSingleStep(100)
        self.points_box.setValue(2000)
        self.points_box.setToolTip(
            "More points resolve narrower fringes. A large arm imbalance needs "
            "a finer sweep or the trace will alias."
        )

        self.power_box = QDoubleSpinBox()
        self.power_box.setRange(0.0, 10_000.0)
        self.power_box.setDecimals(2)
        self.power_box.setSuffix(" mW")
        self.power_box.setValue(20.0)

        self.backend_box = QComboBox()
        for name in available_backends():
            self.backend_box.addItem(name)
        self.backend_box.setToolTip(
            "analytic: closed-form, no extra dependencies.\n"
            "simphony: SiEPIC compact models (needs simphony 0.6.x)."
        )

        for row, (text, widget) in enumerate(
            [
                ("Start", self.start_box),
                ("Stop", self.stop_box),
                ("Points", self.points_box),
                ("Laser power", self.power_box),
                ("Backend", self.backend_box),
            ]
        ):
            grid.addWidget(QLabel(text + ":"), row // 3, (row % 3) * 2)
            grid.addWidget(widget, row // 3, (row % 3) * 2 + 1)

        for box in (self.start_box, self.stop_box):
            box.valueChanged.connect(self._on_circuit_changed)
        self.points_box.valueChanged.connect(self._on_circuit_changed)
        return group

    def _build_monte_carlo_group(self) -> QGroupBox:
        group = QGroupBox("Monte Carlo")
        grid = QGridLayout(group)

        self.mc_enabled = QCheckBox("Run process-variation study")
        self.mc_enabled.setChecked(True)
        grid.addWidget(self.mc_enabled, 0, 0, 1, 2)

        self.mc_runs = QSpinBox()
        self.mc_runs.setRange(1, 5000)
        self.mc_runs.setValue(30)
        grid.addWidget(QLabel("Runs:"), 1, 0)
        grid.addWidget(self.mc_runs, 1, 1)

        self.mc_sigma = QDoubleSpinBox()
        self.mc_sigma.setRange(0.0, 1.0)
        self.mc_sigma.setDecimals(5)
        self.mc_sigma.setSingleStep(0.0005)
        self.mc_sigma.setValue(0.002)
        self.mc_sigma.setToolTip(
            "Standard deviation of the per-waveguide effective-index offset, "
            "standing in for width and thickness variation."
        )
        grid.addWidget(QLabel("sigma n_eff:"), 2, 0)
        grid.addWidget(self.mc_sigma, 2, 1)

        self.mc_seed = QSpinBox()
        self.mc_seed.setRange(0, 2**31 - 1)
        self.mc_seed.setValue(0)
        self.mc_seed.setToolTip("Fixed by default so a run can be reproduced")
        grid.addWidget(QLabel("Seed:"), 3, 0)
        grid.addWidget(self.mc_seed, 3, 1)

        self.mc_enabled.toggled.connect(
            lambda on: [w.setEnabled(on) for w in (self.mc_runs, self.mc_sigma, self.mc_seed)]
        )
        return group

    def _build_status_bar_widget(self) -> QLabel:
        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextFormat(Qt.RichText)
        self.validation_label.setMinimumHeight(38)
        return self.validation_label

    def _build_actions(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setDefault(True)
        self.run_button.setMinimumWidth(140)
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button)

        self.export_figure_button = QPushButton("Save figure...")
        self.export_figure_button.clicked.connect(self._on_save_figure)
        layout.addWidget(self.export_figure_button)

        self.export_data_button = QPushButton("Export data...")
        self.export_data_button.clicked.connect(self._on_export_data)
        layout.addWidget(self.export_data_button)

        for button in (self.export_figure_button, self.export_data_button):
            button.setEnabled(False)

        layout.addStretch(1)
        self.db_toggle = QCheckBox("Plot in dB")
        self.db_toggle.toggled.connect(self._redraw)
        layout.addWidget(self.db_toggle)
        return layout

    def _build_plots(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.sweep_canvas = PlotCanvas()
        self.mc_canvas = PlotCanvas()
        self.tabs.addTab(self.sweep_canvas, "Spectrum")
        self.tabs.addTab(self.mc_canvas, "Monte Carlo")
        return self.tabs

    # ------------------------------------------------------------- presets

    def _apply_preset(self, name: str) -> None:
        slots = _PRESETS.get(name)
        if slots is None:
            return
        for combo, length_box, (component_type, length) in zip(
            self.slot_boxes, self.length_boxes, slots, strict=True
        ):
            combo.blockSignals(True)
            length_box.blockSignals(True)
            combo.setCurrentIndex(combo.findData(component_type))
            if length > 0:
                length_box.setValue(length)
            combo.blockSignals(False)
            length_box.blockSignals(False)
        self._on_circuit_changed()

    # ---------------------------------------------------------- validation

    def _collect_components(self) -> list[Component]:
        components: list[Component] = []
        for combo, length_box in zip(self.slot_boxes, self.length_boxes, strict=True):
            component_type = combo.currentData()
            if component_type is ComponentType.WAVEGUIDE:
                components.append(Component.waveguide(length_box.value() * 1e-6))
            else:
                components.append(Component.of(component_type))
        return components

    def _on_circuit_changed(self, *_args) -> None:
        """Enable the relevant length fields and re-validate the chain."""
        for combo, length_box in zip(self.slot_boxes, self.length_boxes, strict=True):
            length_box.setEnabled(combo.currentData() is ComponentType.WAVEGUIDE)

        try:
            circuit = Circuit.from_components(self._collect_components())
        except CircuitError as exc:
            self._set_validation(str(exc), ok=False)
            self.run_button.setEnabled(False)
            return

        message = f"{circuit.topology.description}"
        if circuit.path_imbalance_m > 0:
            message += f", arm imbalance {circuit.path_imbalance_m * 1e6:g} um"
        self._set_validation(message, ok=True)
        self.run_button.setEnabled(True)

    def _set_validation(self, message: str, *, ok: bool) -> None:
        colour = "#1a7f37" if ok else "#b4232c"
        prefix = "Valid circuit" if ok else "Cannot simulate"
        body = message.replace("\n", "<br>")
        self.validation_label.setText(
            f'<span style="color:{colour}"><b>{prefix}.</b> {body}</span>'
        )

    # -------------------------------------------------------------- running

    def _build_spec(self) -> SweepSpec:
        return SweepSpec(
            start_m=self.start_box.value() * 1e-9,
            stop_m=self.stop_box.value() * 1e-9,
            points=self.points_box.value(),
            laser_power_w=self.power_box.value() * 1e-3,
        )

    def _on_run(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        try:
            circuit = Circuit.from_components(self._collect_components())
            spec = self._build_spec()
        except (CircuitError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot simulate", str(exc))
            return

        mc = (
            MonteCarloSpec(
                runs=self.mc_runs.value(),
                sigma_n_eff=self.mc_sigma.value(),
                seed=self.mc_seed.value(),
            )
            if self.mc_enabled.isChecked()
            else None
        )

        self.run_button.setEnabled(False)
        self.run_button.setText("Running...")
        self.sweep_canvas.show_placeholder("Simulating...")
        if mc is not None:
            self.mc_canvas.show_placeholder("Simulating...")

        self._worker = SimulationWorker(
            circuit, spec, backend=self.backend_box.currentText(), monte_carlo=mc
        )
        self._worker.finished_sweep.connect(self._on_sweep_ready)
        self._worker.finished_monte_carlo.connect(self._on_mc_ready)
        self._worker.failed.connect(self._on_failed)
        self._worker.progress.connect(self.statusBar().showMessage)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self) -> None:
        self.run_button.setEnabled(True)
        self.run_button.setText("Run")
        # Restore the result summary: the worker's last progress message would
        # otherwise be the final thing left in the status bar.
        self._show_summary()

    def _on_sweep_ready(self, result: SweepResult) -> None:
        self._last_sweep = result
        self._draw_sweep()
        self.export_figure_button.setEnabled(True)
        self.export_data_button.setEnabled(True)
        self._show_summary()

    def _show_summary(self) -> None:
        result = self._last_sweep
        if result is None:
            return
        summary = (
            f"IL {result.insertion_loss_db:.2f} dB, ER {result.extinction_ratio_db:.1f} dB"
        )
        fsr = result.free_spectral_range_m()
        if fsr:
            summary += f", FSR {fsr * 1e9:.3f} nm"
        if self._last_mc is not None:
            summary += f" | {len(self._last_mc.runs)} Monte Carlo runs"
        self.statusBar().showMessage(summary)

    def _on_mc_ready(self, result: MonteCarloResult) -> None:
        self._last_mc = result
        self._draw_mc()

    def _on_failed(self, message: str) -> None:
        self.sweep_canvas.show_placeholder("Simulation failed")
        QMessageBox.critical(self, "Simulation failed", message)
        self.statusBar().showMessage("Failed")

    # -------------------------------------------------------------- drawing

    def _draw_sweep(self) -> None:
        if self._last_sweep is None:
            return
        self.sweep_canvas.clear()
        plot_sweep(self._last_sweep, ax=self.sweep_canvas.axes, db=self.db_toggle.isChecked())
        self.sweep_canvas.draw()

    def _draw_mc(self) -> None:
        if self._last_mc is None:
            return
        self.mc_canvas.clear()
        plot_monte_carlo(self._last_mc, ax=self.mc_canvas.axes, db=self.db_toggle.isChecked())
        self.mc_canvas.draw()

    def _redraw(self) -> None:
        self._draw_sweep()
        self._draw_mc()

    # --------------------------------------------------------------- export

    def _on_save_figure(self) -> None:
        canvas = self.tabs.currentWidget()
        if not isinstance(canvas, PlotCanvas):
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", "spectrum.png", "Images (*.png *.pdf *.svg)"
        )
        if path:
            canvas.save(path)
            self.statusBar().showMessage(f"Saved {Path(path).name}")

    def _on_export_data(self) -> None:
        if self._last_sweep is None:
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Export data", "sweep.csv", "CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return
        if path.lower().endswith(".json") or "JSON" in selected:
            if self._last_mc is not None:
                self._last_mc.write_json(path)
            else:
                self._last_sweep.write_json(path)
        else:
            self._last_sweep.write_csv(path)
        self.statusBar().showMessage(f"Exported {Path(path).name}")

    # -------------------------------------------------------------- closing

    def closeEvent(self, event) -> None:
        """Stop a running simulation before the window goes away."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            self._worker.quit()
            self._worker.wait(3000)
        super().closeEvent(event)
