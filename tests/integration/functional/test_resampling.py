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
from rbc_resources import MNI_TEMPLATES

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


def _create_identity_affine() -> Path:
    """Create an ITK-format identity affine transform for testing."""
    out_dir = generate_exec_folder("synthetic_transform")
    mat_file = out_dir / "identity_affine.txt"

    mat_file.write_text(
        "#Insight Transform File V1.0\n"
        "#Transform 0\n"
        "Transform: MatrixOffsetTransformBase_double_3_3\n"
        "Parameters: 1 0 0 0 1 0 0 0 1 0 0 0\n"
        "FixedParameters: 0 0 0\n"
    )

    return mat_file


@pytest.mark.slow
def test_resample_bold_to_template(test_subject: TestSubjectData) -> None:
    """Test resampling on short BOLD timeseries produces output files."""
    from rbc.core.functional import apply_motion_transforms

    template_mni = MNI_TEMPLATES.brain_2mm
    synthetic_wm = _create_synthetic_wm(test_subject.t1w)
    anat_to_template = _create_identity_affine()

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
    preproc_bold = apply_motion_transforms(
        stc_img=stc,
        motion_mat_dir=mc.mat_dir,
        bold_ref=bold_ref,
    )
    masking = bold_masking(
        bold_ref=bold_ref,
        template_mask=MNI_TEMPLATES.brain_mask_2mm,
        template_ref=MNI_TEMPLATES.bold_ref,
    )
    bbr = coregister_bold_to_t1w(
        in_file=masking.skull_stripped_bold,
        reference=test_subject.t1w,
        wm_seg=synthetic_wm,
    )
    template_bold = resample_bold_to_template(
        preproc_bold=preproc_bold,
        bold_to_anat=bbr.out_matrix_file,
        anat_to_template=anat_to_template,
        bold_ref=masking.skull_stripped_bold,
        template=template_mni,
        t1w_brain=test_subject.t1w,
    )
    assert template_bold.exists()

    out_voxel_size = nib.nifti1.load(template_bold).header.get_zooms()[:3]
    in_voxel_size = nib.nifti1.load(stc).header.get_zooms()[:3]
    template_voxel_size = nib.nifti1.load(template_mni).header.get_zooms()[:3]

    # Check that voxel sizes differ between input and output
    assert in_voxel_size != out_voxel_size
    # Check that output voxel size matches template voxel size
    assert out_voxel_size == template_voxel_size
