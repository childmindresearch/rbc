"""Tests for resampling BOLD to template."""

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
    fsl_motion_correction,
    resample_bold_to_template,
    slice_timing_correction,
)
from rbc.core.niwrap import generate_exec_folder
from rbc_resources import REGISTRATION_TEMPLATES

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


def _create_identity_warp(reference: Path) -> Path:
    """Create an identity ANTs/ITK displacement field on the reference grid.

    The new nitransforms-based resampler expects ``anat_to_template`` to
    be a composite displacement field (the format produced by
    ``ants_apply_transforms`` in production), not a `.txt` affine.
    """
    ref_img = nib.nifti1.load(reference)
    warp = np.zeros((*ref_img.shape[:3], 1, 3), dtype=np.float32)
    img = nib.Nifti1Image(warp, ref_img.affine)
    img.header.set_intent("vector")
    out_dir = generate_exec_folder("synthetic_transform")
    out_path = out_dir / "identity_warp.nii.gz"
    nib.save(img, out_path)
    return out_path


@pytest.mark.slow
def test_resample_bold_to_template(test_subject: TestSubjectData) -> None:
    """Test single-step resampling of STC BOLD to template space."""
    template_mni = REGISTRATION_TEMPLATES.brain_2mm
    synthetic_wm = _create_synthetic_wm(test_subject.t1w)
    anat_to_template = _create_identity_warp(template_mni)

    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..49]"
        ),
        expression="a",
        prefix="test_bold.nii.gz",
    )
    assert truncated.output_file is not None

    bold_ref = extract_motion_reference(in_file=truncated.output_file)
    mc = fsl_motion_correction(in_file=truncated.output_file, ref_file=bold_ref)
    stc = slice_timing_correction(in_file=truncated.output_file)
    masking = bold_masking(
        bold_ref=bold_ref,
        template_mask=REGISTRATION_TEMPLATES.brain_mask_2mm,
        template_ref=REGISTRATION_TEMPLATES.bold_ref,
    )
    bbr = coregister_bold_to_t1w(
        in_file=masking.skull_stripped_bold,
        reference=test_subject.t1w,
        wm_seg=synthetic_wm,
    )
    template_bold = resample_bold_to_template(
        stc_bold=stc,
        motion_mat_dir=mc.mat_dir,
        bold_to_anat=bbr.out_matrix_file,
        anat_to_template=anat_to_template,
        bold_ref=masking.skull_stripped_bold,
        template=template_mni,
        t1w_brain=test_subject.t1w,
    )
    assert template_bold.exists()

    out_img = nib.nifti1.load(template_bold)
    in_img = nib.nifti1.load(stc)
    template_img = nib.nifti1.load(template_mni)

    out_voxel_size = out_img.header.get_zooms()[:3]
    in_voxel_size = in_img.header.get_zooms()[:3]
    template_voxel_size = template_img.header.get_zooms()[:3]

    # Check that voxel sizes differ between input and output
    assert in_voxel_size != out_voxel_size
    # Check that output voxel size matches template voxel size
    assert out_voxel_size == template_voxel_size
    # Check that TR is preserved from the source BOLD (not overwritten by the
    # template's pixdim[4])
    assert out_img.header.get_zooms()[3] == in_img.header.get_zooms()[3]
