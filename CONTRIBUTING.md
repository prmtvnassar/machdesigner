# Contributing

## Setup

```bash
git clone https://github.com/prmtv-mind/machdesigner
cd machdesigner
python -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]'
pytest
```

On a headless Linux machine, Qt needs an offscreen platform:

```bash
QT_QPA_PLATFORM=offscreen pytest
```

## Before opening a pull request

```bash
ruff check src tests
ruff format src tests
mypy
pytest
```

CI runs all four across Python 3.10-3.13, plus a job that installs *without*
the optional extras to confirm the core library still works when Qt and
simphony are absent.

## How the code is organised

The physics is a library; the GUI and CLI are thin front ends over it. Keep it
that way — anything that cannot be tested without a display belongs in
`gui/`, and nothing in `gui/` should contain physics.

```
components.py   compact models for each component
netlist.py      circuit description, validation, topology recognition
sweep.py        input specifications
results.py      output metrics and export
simulate.py     high-level entry points
plotting.py     figure builders (never call plt.show or plt.savefig here)
cli.py          command line
backends/       one module per simulation engine
gui/            Qt front end
```

## Adding a topology

1. Add its component signature to `_SIGNATURES` in `netlist.py`, and a case to
   `_suggest()` so a near-miss produces a useful message.
2. Add a solver branch in `backends/analytic.py`, and in
   `backends/simphony_backend.py` if simphony can build it.
3. Add a physics test that checks the new solver against an independently
   derived relation, not against a recorded output.
4. Add it to the preset list in `gui/main_window.py` and to the table in the
   README.

The GUI and CLI need no other changes; both read the supported set from the
netlist layer.

## Adding a backend

Implement the `Backend` protocol in `backends/base.py` — `name`, `sweep()`,
and `monte_carlo()` — then register it in `backends/__init__.py`. Raise
`BackendUnavailableError` with an install hint if optional dependencies are
missing, rather than letting an `ImportError` escape.

New backends should pass the cross-agreement tests in `tests/test_backends.py`.
Two engines that make different assumptions but agree on free spectral range
is the strongest signal we have that both are right.

## Tests

Physics tests validate against derived relations, not recorded output. A test
that asserts `transmission[42] == 0.2314` tells you a number changed; a test
that asserts `FSR == lam^2 / (n_g dL)` tells you whether the change was wrong.
Prefer the second.

Tests requiring optional dependencies should skip cleanly:

```python
pytest.importorskip("PyQt5")

requires_simphony = pytest.mark.skipif(
    not simphony_available(), reason="simphony 0.6.x is not installed"
)
```

## Model parameters

Default parameters in `components.py` are *nominal* SOI values, not a
calibrated PDK. If you change one, say in the docstring where the value comes
from and update the table in `docs/physics.md`. If a change makes the model
more accurate in a way a test can detect, add that test.

Do not quote a parameter to more significant figures than its source supports.
