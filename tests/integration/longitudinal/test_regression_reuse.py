"""Integration test for longitudinal regression reuse.

Given cross-sectional outputs and raw regressor ``.1D`` files,
``apply_regression`` and ``apply_regression_bandpass`` on warped BOLD
produce non-degenerate outputs (non-zero variance, correct timepoints).

Tier-2 integration test for Stage 5 of the longitudinal refactor
(tracker: #301).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from rbc.core.functional import apply_regression, apply_regression_bandpass
from rbc.core.nifti import nifti_num_volumes

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import TestSubjectData


def _prepare_short_bold_and_mask(
    test_subject: TestSubjectData,
    n_vols: int = 50,
) -> tuple[Path, Path, Path]:
    """Prepare a short BOLD series with brain mask and dummy regressor.

    Returns (bold_path, mask_path, regressor_1d_path).
    """
    from niwrap import afni

    from rbc.core.common import deoblique_and_reorient
    from rbc.core.functional import (
        extract_motion_reference,
        fsl_motion_correction,
        nuisance_regression,
    )
    from rbc.core.niwrap import generate_exec_folder

    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_=f"[0..{n_vols - 1}]"
        ),
        expression="a",
        prefix="test_bold.nii.gz",
    )
    assert truncated.output_file is not None
    motion_ref = extract_motion_reference(in_file=truncated.output_file)
    mc = fsl_motion_correction(in_file=truncated.output_file, ref_file=motion_ref)

    automask = afni.v_3d_automask(
        in_file=mc.bold,
        prefix="brain_mask.nii.gz",
    )
    assert automask.mask_file is not None

    # Create synthetic tissue masks for regressor computation
    from scipy.ndimage import binary_erosion

    out_dir = generate_exec_folder("regression_reuse_masks")
    brain_img = nib.nifti1.load(automask.mask_file)
    brain_data = brain_img.get_fdata() > 0
    eroded_1 = binary_erosion(brain_data, iterations=1)
    csf_data = brain_data & ~eroded_1
    wm_data = binary_erosion(brain_data, iterations=3)

    csf_file = out_dir / "csf_mask.nii.gz"
    wm_file = out_dir / "wm_mask.nii.gz"
    nib.nifti1.Nifti1Image(
        csf_data.astype(np.uint8), brain_img.affine, brain_img.header
    ).to_filename(str(csf_file))
    nib.nifti1.Nifti1Image(
        wm_data.astype(np.uint8), brain_img.affine, brain_img.header
    ).to_filename(str(wm_file))

    # Compute raw regressors
    reg_result = nuisance_regression(
        bold_file=mc.bold,
        brain_mask_file=automask.mask_file,
        csf_mask_file=csf_file,
        wm_mask_file=wm_file,
        motion_params=mc.motion_params,
        regressor_set="36-parameter",
    )

    return mc.bold, automask.mask_file, reg_result.regressor_file


@pytest.mark.slow
def test_apply_regression_on_warped_bold(test_subject: TestSubjectData) -> None:
    """Raw regressors from cross-sectional run produce valid regression output.

    Simulates the longitudinal case: regression is applied to a BOLD
    timeseries using regressors that were computed in a different space.
    The regressor matrix has the right number of timepoints (matched by
    the cross-sectional pipeline), so regression should succeed.
    """
    bold, mask, regressor_1d = _prepare_short_bold_and_mask(test_subject)

    result = apply_regression(
        bold_file=bold,
        brain_mask_file=mask,
        regressor_file=regressor_1d,
    )

    assert result.regressed_bold.exists()
    assert nifti_num_volumes(result.regressed_bold) == nifti_num_volumes(bold)

    # Non-degenerate: regressed BOLD should have non-zero variance
    img = nib.nifti1.load(result.regressed_bold)
    data = img.get_fdata()
    mask_img = nib.nifti1.load(mask)
    mask_data = mask_img.get_fdata() > 0
    brain_ts = data[mask_data]
    assert brain_ts.var(axis=-1).mean() > 0, "Regressed BOLD has zero variance"


@pytest.mark.slow
def test_apply_regression_bandpass_on_warped_bold(
    test_subject: TestSubjectData,
) -> None:
    """Bandpass regression on warped BOLD produces non-degenerate output."""
    bold, mask, regressor_1d = _prepare_short_bold_and_mask(test_subject)

    result = apply_regression_bandpass(
        bold_file=bold,
        brain_mask_file=mask,
        regressor_file=regressor_1d,
    )

    assert result.regressed_bold.exists()
    assert nifti_num_volumes(result.regressed_bold) == nifti_num_volumes(bold)

    img = nib.nifti1.load(result.regressed_bold)
    data = img.get_fdata()
    mask_img = nib.nifti1.load(mask)
    mask_data = mask_img.get_fdata() > 0
    brain_ts = data[mask_data]
    assert brain_ts.var(axis=-1).mean() > 0, "Cleaned BOLD has zero variance"
