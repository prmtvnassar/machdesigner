"""Shared fixtures."""

from __future__ import annotations

import matplotlib
import pytest

# Headless by default; individual GUI tests opt into the Qt backend.
matplotlib.use("Agg")

from machdesigner.components import (
    PDK,
    GratingCouplerModel,
    WaveguideModel,
    YBranchModel,
)


@pytest.fixture
def ideal_pdk() -> PDK:
    """A lossless, spectrally flat kit.

    Strips away every loss term so simulated transmission can be compared
    directly against the textbook interferometer relation. The grating-coupler
    bandwidth is made enormous so its Gaussian envelope is flat (within 1e-12)
    across any realistic sweep.
    """
    return PDK(
        waveguide=WaveguideModel(loss_db_per_cm=0.0),
        grating_coupler=GratingCouplerModel(peak_insertion_loss_db=0.0, bandwidth_3db_m=1.0),
        y_branch=YBranchModel(excess_loss_db=0.0),
    )


@pytest.fixture
def default_pdk() -> PDK:
    return PDK.default()
