"""Circuit validation and topology recognition.

The original tool accepted exactly one circuit and rejected everything else
with the same opaque message. These tests pin down both halves of the
replacement: that valid circuits are recognised, and that invalid ones produce
a diagnostic naming the actual problem.
"""

from __future__ import annotations

import pytest

from machdesigner import Circuit, CircuitError, Topology
from machdesigner.components import Component, ComponentType

GC_IN = Component.of(ComponentType.GC_INPUT)
GC_OUT = Component.of(ComponentType.GC_OUTPUT)
SPLIT = Component.of(ComponentType.Y_SPLITTER)
COMBINE = Component.of(ComponentType.Y_COMBINER)
NONE = Component.of(ComponentType.NONE)


def wg(length_um: float) -> Component:
    return Component.waveguide(length_um * 1e-6)


# ------------------------------------------------------------------- valid


def test_recognises_the_mzi():
    circuit = Circuit.from_components([GC_IN, SPLIT, wg(150), wg(50), COMBINE, GC_OUT])
    assert circuit.topology is Topology.MZI
    assert circuit.arm_lengths_m == pytest.approx((150e-6, 50e-6))
    assert circuit.path_imbalance_m == pytest.approx(100e-6)


def test_recognises_the_straight_link():
    circuit = Circuit.from_components([GC_IN, wg(100), GC_OUT])
    assert circuit.topology is Topology.STRAIGHT
    assert circuit.arm_lengths_m == pytest.approx((100e-6,))
    assert circuit.path_imbalance_m == 0.0


def test_trailing_empty_slots_are_ignored():
    """The GUI always shows six slots; a short circuit leaves some empty."""
    circuit = Circuit.from_components([GC_IN, wg(100), GC_OUT, NONE, NONE, NONE])
    assert circuit.topology is Topology.STRAIGHT
    assert len(circuit.components) == 3


def test_arm_order_does_not_change_the_imbalance():
    forward = Circuit.from_components([GC_IN, SPLIT, wg(150), wg(50), COMBINE, GC_OUT])
    reversed_ = Circuit.from_components([GC_IN, SPLIT, wg(50), wg(150), COMBINE, GC_OUT])
    assert forward.path_imbalance_m == reversed_.path_imbalance_m


def test_mzi_helper_matches_manual_construction():
    manual = Circuit.from_components([GC_IN, SPLIT, wg(150), wg(50), COMBINE, GC_OUT])
    helper = Circuit.mzi(150e-6, 50e-6)
    assert helper.topology is manual.topology
    assert [c.type for c in helper.components] == [c.type for c in manual.components]
    assert helper.arm_lengths_m == pytest.approx(manual.arm_lengths_m)


# ----------------------------------------------------------------- invalid


def test_empty_circuit_is_rejected():
    with pytest.raises(CircuitError, match="empty"):
        Circuit.from_components([])


def test_all_empty_slots_are_rejected():
    with pytest.raises(CircuitError, match="empty"):
        Circuit.from_components([NONE] * 6)


def test_circuit_must_start_with_an_input_coupler():
    with pytest.raises(CircuitError, match="must start with an input grating coupler"):
        Circuit.from_components([wg(100), GC_OUT])


def test_circuit_must_end_with_an_output_coupler():
    with pytest.raises(CircuitError, match="must end with an output grating coupler"):
        Circuit.from_components([GC_IN, wg(100)])


def test_interior_gap_is_reported_with_its_slot_number():
    with pytest.raises(CircuitError, match="slot 2 is empty"):
        Circuit.from_components([GC_IN, NONE, wg(100), GC_OUT])


def test_splitter_without_combiner_is_explained():
    with pytest.raises(CircuitError, match="matching Y-combiner"):
        Circuit.from_components([GC_IN, SPLIT, wg(150), wg(50), GC_OUT])


def test_combiner_without_splitter_is_explained():
    with pytest.raises(CircuitError, match="preceding Y-splitter"):
        Circuit.from_components([GC_IN, wg(150), wg(50), COMBINE, GC_OUT])


def test_wrong_arm_count_names_the_count():
    with pytest.raises(CircuitError, match="exactly two arm waveguides"):
        Circuit.from_components([GC_IN, SPLIT, wg(1), wg(2), wg(3), COMBINE, GC_OUT])


def test_cascaded_interferometers_are_reported_as_unsupported():
    with pytest.raises(CircuitError, match="Nested or cascaded"):
        Circuit.from_components([GC_IN, SPLIT, SPLIT, wg(1), wg(2), COMBINE, COMBINE, GC_OUT])


def test_error_carries_a_suggestion():
    with pytest.raises(CircuitError) as info:
        Circuit.from_components([wg(100), GC_OUT])
    assert info.value.suggestion
    assert "Grating coupler (input)" in info.value.suggestion


# ------------------------------------------------------------------ legacy


def test_legacy_dropdown_indices_still_build_a_circuit():
    """The original GUI encoded components as integers 0-5."""
    circuit = Circuit.from_legacy_indices([1, 3, 5, 5, 4, 2], [150e-6, 50e-6])
    assert circuit.topology is Topology.MZI
    assert circuit.arm_lengths_m == pytest.approx((150e-6, 50e-6))


def test_legacy_indices_need_one_length_per_waveguide():
    with pytest.raises(CircuitError, match="more waveguide slots than lengths"):
        Circuit.from_legacy_indices([1, 3, 5, 5, 4, 2], [150e-6])


# -------------------------------------------------------------- components


def test_waveguide_requires_a_positive_length():
    with pytest.raises(ValueError, match="must be positive"):
        Component.waveguide(0.0)
    with pytest.raises(ValueError, match="must be positive"):
        Component.waveguide(-1e-6)


def test_component_lookup_is_case_and_separator_insensitive():
    assert ComponentType.from_label("GC Input") is ComponentType.GC_INPUT
    assert ComponentType.from_label("y-splitter") is ComponentType.Y_SPLITTER
    assert ComponentType.from_label("WG") is ComponentType.WAVEGUIDE


def test_unknown_component_name_is_rejected():
    with pytest.raises(ValueError, match="unknown component"):
        ComponentType.from_label("ring resonator")


def test_legacy_indices_match_the_enum():
    """GUI dropdown order must keep matching the enum values."""
    assert [ComponentType(i).name for i in range(6)] == [
        "NONE",
        "GC_INPUT",
        "GC_OUTPUT",
        "Y_SPLITTER",
        "Y_COMBINER",
        "WAVEGUIDE",
    ]
