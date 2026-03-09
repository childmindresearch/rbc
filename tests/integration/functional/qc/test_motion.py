"""Integration tests for motion QC metrics using real MCFLIRT outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rbc.core.qc.motion import (
    framewise_displacement_jenkinson,
    framewise_displacement_power,
    motion_qc_metrics,
    rms_motion,
)

if TYPE_CHECKING:
    from tests.integration.functional.qc.conftest import MotionCorrectedBOLD


@pytest.mark.slow
def test_motion_qc_from_mcflirt(motion_corrected_bold: MotionCorrectedBOLD) -> None:
    """Compute motion QC metrics from real MCFLIRT outputs and sanity-check."""
    mc = motion_corrected_bold.mc

    # Read MCFLIRT outputs
    rms_values = np.loadtxt(mc.rms_rel)
    motion_params = np.loadtxt(mc.motion_params)

    # --- FD-Jenkinson ---
    fd_j = framewise_displacement_jenkinson(rms_values)
    assert len(fd_j) == motion_params.shape[0]
    assert fd_j[0] == 0.0
    assert np.all(fd_j >= 0)

    # --- FD-Power ---
    fd_p = framewise_displacement_power(motion_params)
    assert len(fd_p) == motion_params.shape[0]
    assert fd_p[0] == 0.0
    assert np.all(fd_p >= 0)

    # --- RMS motion ---
    mean_rms, max_rms = rms_motion(motion_params)
    assert mean_rms >= 0
    assert max_rms >= mean_rms

    # --- Combined metrics ---
    m = motion_qc_metrics(rms_values, motion_params)
    assert m.mean_fd >= 0
    assert m.rel_means_rms_motion >= 0
    assert m.rel_max_rms_motion >= m.rel_means_rms_motion
    assert m.n_vol_censored >= 0
    assert m.n_vol_censored <= motion_params.shape[0]
