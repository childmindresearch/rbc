"""Integration tests for motion QC metrics using real MCFLIRT outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from niwrap import afni

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
)
from rbc.core.qc.motion import (
    framewise_displacement_jenkinson,
    framewise_displacement_power,
    motion_qc_metrics,
    rms_motion,
)

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_motion_qc_from_mcflirt(test_subject: TestSubjectData) -> None:
    """Compute motion QC metrics from real MCFLIRT outputs and sanity-check."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..9]"
        ),
        expression="a",
        prefix="qc_test_10vols.nii.gz",
    )

    ref = extract_motion_reference(in_file=truncated.output_file)
    mc = fsl_motion_correction(
        in_file=truncated.output_file,
        ref_file=ref.output_file,
    )

    # Read MCFLIRT outputs
    rms_values = np.loadtxt(mc.rms_rel)
    motion_params = np.loadtxt(mc.par)

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
