"""Tests for coregistration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest
from niwrap import afni
from scipy.ndimage import binary_erosion

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    bold_masking,
    coregister_bold_to_t1w,
    extract_motion_reference,
    slice_timing_correction,
    truncate_trs,
)
from rbc.core.niwrap import generate_exec_folder
from rbc.core.resources import MNI_TEMPLATES

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import TestSubjectData


def _create_synthetic_wm(t1w: Path) -> Path:
    """Create synthetic WM mask from T1w for testing.

    - Brain mask: AFNI 3dAutomask
    - WM mask: inner core of brain mask (eroded by 3 voxels)
    """
    out_dir = generate_exec_folder("synthetic_wm")

    automask = afni.v_3d_automask(
        in_file=t1w,
        prefix="t1w_brain_mask.nii.gz",
    )

    mask_img = nib.nifti1.load(automask.mask_file)
    mask_data = mask_img.get_fdata() > 0

    wm_file = out_dir / "wm_bbr_mask.nii.gz"
    nib.nifti1.Nifti1Image(
        binary_erosion(mask_data, iterations=3).astype(np.uint8),
        mask_img.affine,
        mask_img.header,
    ).to_filename(str(wm_file))

    return wm_file


@pytest.mark.slow
def test_coregistration(test_subject: TestSubjectData) -> None:
    """Test coregistration produces output files."""
    # Anatomical
    reoriented_t1w = deoblique_and_reorient(in_file=test_subject.t1w)
    synthetic_wm = _create_synthetic_wm(reoriented_t1w.out_file)

    # Functional
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = truncate_trs(in_file=reoriented.out_file, start_tr=2)
    stc = slice_timing_correction(in_file=truncated.output_file)
    bold_ref = extract_motion_reference(in_file=stc.out_file)
    masking = bold_masking(
        bold_ref=bold_ref,
        template_mask=MNI_TEMPLATES.brain_mask_2mm,
        template_ref=MNI_TEMPLATES.bold_ref,
    )
    coregistration = coregister_bold_to_t1w(
        in_file=masking.skull_stripped_bold,
        reference=reoriented_t1w.out_file,
        wm_seg=synthetic_wm,
    )
    assert coregistration.out_matrix_file.exists()
    assert coregistration.out_file.exists()
