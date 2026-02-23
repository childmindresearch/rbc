"""Integration tests for DVARS metrics using real BOLD data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from rbc.core.qc.dvars import dvars, dvars_qc_metrics, motion_dvars_correlation
from rbc.core.qc.motion import framewise_displacement_jenkinson

if TYPE_CHECKING:
    from tests.integration.functional.qc.conftest import MotionCorrectedBOLD


@pytest.mark.slow
def test_dvars_from_bold(motion_corrected_bold: MotionCorrectedBOLD) -> None:
    """Compute DVARS from real BOLD data and sanity-check."""
    mc = motion_corrected_bold.mc
    bold_data = motion_corrected_bold.bold_data
    mask = motion_corrected_bold.mask

    n_vols = bold_data.shape[3]

    # --- DVARS ---
    dv = dvars(bold_data, mask)
    assert len(dv) == n_vols
    assert dv[0] == 0.0
    assert np.all(dv >= 0)

    # --- Motion-DVARS correlation ---
    rms_values = np.loadtxt(mc.rms_rel)
    fd = framewise_displacement_jenkinson(rms_values)
    corr = motion_dvars_correlation(dv, fd)
    assert -1.0 <= corr <= 1.0

    # --- Combined metrics ---
    m = dvars_qc_metrics(bold_data, mask, fd)
    assert m.mean_dvars >= 0
    assert -1.0 <= m.motion_dvars_corr <= 1.0
