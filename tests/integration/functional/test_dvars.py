"""Integration tests for DVARS metrics using real BOLD data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest
from niwrap import afni

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
)
from rbc.core.qc.dvars import dvars, dvars_qc_metrics, motion_dvars_correlation
from rbc.core.qc.motion import framewise_displacement_jenkinson

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_dvars_from_bold(test_subject: TestSubjectData) -> None:
    """Compute DVARS from real BOLD data and sanity-check."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..9]"
        ),
        expression="a",
        prefix="dvars_test_10vols.nii.gz",
    )

    ref = extract_motion_reference(in_file=truncated.output_file)
    mc = fsl_motion_correction(
        in_file=truncated.output_file,
        ref_file=ref.output_file,
    )

    # Load motion-corrected BOLD and create a simple mask
    bold_img = nib.nifti1.load(mc.bold.with_suffix(".nii.gz"))
    bold_data = bold_img.get_fdata()
    mask = np.mean(bold_data, axis=3) > 0

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
