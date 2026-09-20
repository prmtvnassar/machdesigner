"""Circuit description, topology recognition, and validation.

The original MachDesigner accepted exactly one circuit, enforced by comparing
raw dropdown indices against hard-coded integers. Anything else produced the
message "There is something wrong with the connection!" with no indication of
what was wrong.

This module replaces that with a structural description. A :class:`Circuit` is
an ordered chain of components; validation walks the chain, recognises the
topology, and reports a specific diagnostic when it cannot. Adding a new
supported topology means adding a rule here, not editing the GUI.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from .components import Component, ComponentType, describe

__all__ = [
    "Circuit",
    "CircuitError",
    "Topology",
]


class Topology(Enum):
    """A circuit shape the simulator knows how to solve."""

    STRAIGHT = "straight"
    """Grating coupler, one waveguide, grating coupler. No interference."""

    MZI = "mzi"
    """Mach-Zehnder interferometer: splitter, two arms, combiner."""

    @property
    def description(self) -> str:
        return {
            Topology.STRAIGHT: "straight waveguide link",
            Topology.MZI: "Mach-Zehnder interferometer",
        }[self]


class CircuitError(ValueError):
    """Raised when a component chain is not a circuit this tool can solve.

    Carries a human-readable explanation plus, where possible, a concrete
    suggestion; the GUI surfaces both instead of a generic failure message.
    """

    def __init__(self, message: str, *, suggestion: str | None = None) -> None:
        self.suggestion = suggestion
        super().__init__(message if suggestion is None else f"{message}\n{suggestion}")


#: Canonical component order for each supported topology.
_SIGNATURES: dict[tuple[ComponentType, ...], Topology] = {
    (
        ComponentType.GC_INPUT,
        ComponentType.WAVEGUIDE,
        ComponentType.GC_OUTPUT,
    ): Topology.STRAIGHT,
    (
        ComponentType.GC_INPUT,
        ComponentType.Y_SPLITTER,
        ComponentType.WAVEGUIDE,
        ComponentType.WAVEGUIDE,
        ComponentType.Y_COMBINER,
        ComponentType.GC_OUTPUT,
    ): Topology.MZI,
}


@dataclass(frozen=True)
class Circuit:
    """A validated chain of photonic components.

    Construct with :meth:`from_components`, :meth:`mzi`, or
    :meth:`from_legacy_indices`; the constructor does not validate on its own
    so that callers can inspect an invalid chain before deciding what to do.
    """

    components: tuple[Component, ...]
    topology: Topology

    # ----------------------------------------------------------------- build

    @classmethod
    def from_components(cls, components: Iterable[Component]) -> Circuit:
        """Validate a component chain and identify its topology.

        Raises
        ------
        CircuitError
            If the chain is empty, contains gaps, or does not match a
            supported topology. The message names the specific problem.
        """
        chain = tuple(components)
        _reject_empty(chain)
        _reject_interior_gaps(chain)
        chain = tuple(c for c in chain if c.type is not ComponentType.NONE)
        _reject_empty(chain)
        _check_endpoints(chain)

        signature = tuple(c.type for c in chain)
        topology = _SIGNATURES.get(signature)
        if topology is None:
            raise CircuitError(
                f"unsupported circuit: {describe(chain)}",
                suggestion=_suggest(signature),
            )
        return cls(components=chain, topology=topology)

    @classmethod
    def mzi(cls, long_arm_m: float, short_arm_m: float) -> Circuit:
        """Build the canonical unbalanced MZI used by the GUI defaults."""
        return cls.from_components(
            [
                Component.of(ComponentType.GC_INPUT),
                Component.of(ComponentType.Y_SPLITTER),
                Component.waveguide(long_arm_m),
                Component.waveguide(short_arm_m),
                Component.of(ComponentType.Y_COMBINER),
                Component.of(ComponentType.GC_OUTPUT),
            ]
        )

    @classmethod
    def straight(cls, length_m: float) -> Circuit:
        """Build a single-waveguide link between two grating couplers."""
        return cls.from_components(
            [
                Component.of(ComponentType.GC_INPUT),
                Component.waveguide(length_m),
                Component.of(ComponentType.GC_OUTPUT),
            ]
        )

    @classmethod
    def from_legacy_indices(cls, indices: Sequence[int], lengths_m: Sequence[float]) -> Circuit:
        """Build from raw GUI dropdown indices and a queue of waveguide lengths.

        ``indices`` uses the original MachDesigner encoding (0 = None,
        1 = GC input, ... 5 = waveguide). Each waveguide slot consumes one
        entry from ``lengths_m`` in order.
        """
        lengths = list(lengths_m)
        components: list[Component] = []
        for index in indices:
            type_ = ComponentType(index)
            if type_ is ComponentType.WAVEGUIDE:
                if not lengths:
                    raise CircuitError(
                        "more waveguide slots than lengths were supplied",
                        suggestion="Provide one length per waveguide slot.",
                    )
                components.append(Component.waveguide(lengths.pop(0)))
            else:
                components.append(Component.of(type_))
        return cls.from_components(components)

    # ------------------------------------------------------------ accessors

    @property
    def waveguides(self) -> tuple[Component, ...]:
        """The waveguide components, in chain order."""
        return tuple(c for c in self.components if c.type is ComponentType.WAVEGUIDE)

    @property
    def arm_lengths_m(self) -> tuple[float, ...]:
        """Waveguide lengths in metres, in chain order."""
        return tuple(c.length_m for c in self.waveguides)

    @property
    def path_imbalance_m(self) -> float:
        """Optical path difference between the two MZI arms.

        Zero for topologies without two arms. A balanced MZI (imbalance of
        zero) has no fringes, which is a legitimate configuration rather than
        an error.
        """
        if self.topology is not Topology.MZI:
            return 0.0
        long_arm, short_arm = self.arm_lengths_m
        return abs(long_arm - short_arm)

    def __str__(self) -> str:
        return f"{self.topology.description}: {describe(self.components)}"


# --------------------------------------------------------------- validation


def _reject_empty(chain: Sequence[Component]) -> None:
    if not chain:
        raise CircuitError(
            "the circuit is empty",
            suggestion="Select at least an input coupler, a waveguide, and an output coupler.",
        )


def _reject_interior_gaps(chain: Sequence[Component]) -> None:
    """Reject an empty slot sitting between two real components.

    Trailing empty slots are fine -- the GUI always shows six of them -- but a
    gap in the middle means the user left a hole in the chain, which is almost
    always a mistake rather than an intentional shorter circuit.
    """
    types = [c.type for c in chain]
    filled = [i for i, t in enumerate(types) if t is not ComponentType.NONE]
    if not filled:
        return
    first, last = filled[0], filled[-1]
    gaps = [i + 1 for i in range(first, last) if types[i] is ComponentType.NONE]
    if gaps:
        positions = ", ".join(str(g) for g in gaps)
        raise CircuitError(
            f"slot {positions} is empty but sits between two components",
            suggestion="Fill the gap, or move the remaining components left so "
            "the chain is contiguous.",
        )


def _check_endpoints(chain: Sequence[Component]) -> None:
    if chain[0].type is not ComponentType.GC_INPUT:
        raise CircuitError(
            f"a circuit must start with an input grating coupler, "
            f"but slot 1 holds {chain[0].label}",
            suggestion="Set the first component to 'Grating coupler (input)'.",
        )
    if chain[-1].type is not ComponentType.GC_OUTPUT:
        raise CircuitError(
            f"a circuit must end with an output grating coupler, "
            f"but the last component is {chain[-1].label}",
            suggestion="Set the final component to 'Grating coupler (output)'.",
        )


def _suggest(signature: tuple[ComponentType, ...]) -> str:
    """Offer the most useful hint we can for an unrecognised chain."""
    counts = {t: signature.count(t) for t in set(signature)}
    splitters = counts.get(ComponentType.Y_SPLITTER, 0)
    combiners = counts.get(ComponentType.Y_COMBINER, 0)
    waveguides = counts.get(ComponentType.WAVEGUIDE, 0)

    if splitters and not combiners:
        return "A Y-splitter needs a matching Y-combiner to close the interferometer."
    if combiners and not splitters:
        return "A Y-combiner needs a preceding Y-splitter to feed both arms."
    if splitters and combiners and waveguides != 2:
        return (
            f"An interferometer needs exactly two arm waveguides between the "
            f"splitter and combiner; this chain has {waveguides}."
        )
    if splitters > 1 or combiners > 1:
        return "Nested or cascaded interferometers are not supported yet."
    return (
        "Supported circuits are:\n"
        "  - GC input -> waveguide -> GC output\n"
        "  - GC input -> Y-splitter -> waveguide -> waveguide -> Y-combiner -> GC output"
    )
