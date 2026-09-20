"""Simulation backends and the registry used to select one."""

from __future__ import annotations

from .analytic import AnalyticBackend
from .base import Backend, BackendUnavailableError
from .simphony_backend import SimphonyBackend, simphony_available

__all__ = [
    "BACKEND_NAMES",
    "AnalyticBackend",
    "Backend",
    "BackendUnavailableError",
    "SimphonyBackend",
    "available_backends",
    "get_backend",
    "simphony_available",
]

#: Every backend name the CLI and GUI accept, plus the ``auto`` alias.
BACKEND_NAMES = ("auto", "analytic", "simphony")


def available_backends() -> tuple[str, ...]:
    """Backend names that can actually run in this environment."""
    names = ["analytic"]
    if simphony_available():
        names.append("simphony")
    return tuple(names)


def get_backend(name: str = "auto", **kwargs) -> Backend:
    """Instantiate a backend by name.

    ``auto`` resolves to ``analytic``. The analytic engine is the default
    because it needs no optional dependencies and is fast enough to drive a
    GUI interactively; ask for ``simphony`` explicitly when you want the
    SiEPIC compact models.

    Raises
    ------
    ValueError
        If ``name`` is not a known backend.
    BackendUnavailableError
        If the named backend's dependencies are missing.
    """
    key = (name or "auto").strip().lower()
    if key == "auto":
        key = "analytic"
    if key == "analytic":
        return AnalyticBackend(**kwargs)
    if key == "simphony":
        return SimphonyBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKEND_NAMES)}")
