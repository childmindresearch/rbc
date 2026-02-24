"""Integration tests for nuisance regression."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
from niwrap import afni
from scipy.ndimage import binary_erosion

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
    nuisance_regression,
)
from rbc.core.nifti import nifti_num_volumes
from rbc.core.niwrap import generate_exec_folder

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import TestSubjectData


class _PreparedData(NamedTuple):
    bold: Path
    par: Path
    brain_mask: Path
    csf_mask: Path
    wm_mask: Path


def _create_synthetic_masks(brain_mask_file: Path) -> tuple[Path, Path, Path]:
    """Create synthetic CSF and WM masks from a brain mask for testing.

    Generates tissue masks by subdividing the brain mask into shells:
    - CSF: outer rim of the brain mask (1-voxel thick shell)
    - WM: inner core of the brain mask (eroded by 3 voxels)

    These are not anatomically accurate but provide valid masks for
    testing the nuisance regression pipeline.
    """
    import nibabel as nib

    out_dir = generate_exec_folder("synthetic_masks")

    brain_img = nib.nifti1.load(brain_mask_file)
    brain_data = brain_img.get_fdata() > 0

    # CSF: outer rim (brain minus brain eroded by 1 voxel)
    eroded_1 = binary_erosion(brain_data, iterations=1)
    csf_data = brain_data & ~eroded_1

    # WM: deep interior (brain eroded by 3 voxels)
    wm_data = binary_erosion(brain_data, iterations=3)

    csf_file = out_dir / "csf_mask.nii.gz"
    wm_file = out_dir / "wm_mask.nii.gz"

    nib.nifti1.Nifti1Image(
        csf_data.astype(np.uint8), brain_img.affine, brain_img.header
    ).to_filename(str(csf_file))
    nib.nifti1.Nifti1Image(
        wm_data.astype(np.uint8), brain_img.affine, brain_img.header
    ).to_filename(str(wm_file))

    return brain_mask_file, csf_file, wm_file


def _prepare_bold_and_masks(
    test_subject: TestSubjectData,
    n_vols: int = 50,
) -> _PreparedData:
    """Prepare a short BOLD series, motion params, and masks.

    Creates a brain mask from the BOLD data using AFNI 3dAutomask, then
    derives synthetic CSF/WM masks from it.
    """
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_=f"[0..{n_vols - 1}]"
        ),
        expression="a",
        prefix="test_bold.nii.gz",
    )
    motion_ref = extract_motion_reference(in_file=truncated.output_file)
    mc = fsl_motion_correction(in_file=truncated.output_file, ref_file=motion_ref)
    bold_file = mc.bold.with_suffix(".nii.gz")

    # Create brain mask from BOLD using 3dAutomask
    automask = afni.v_3d_automask(
        in_file=bold_file,
        prefix="brain_mask.nii.gz",
    )

    # Create synthetic tissue masks from brain mask
    brain_mask, csf_mask, wm_mask = _create_synthetic_masks(automask.mask_file)

    return _PreparedData(
        bold=bold_file,
        par=mc.par,
        brain_mask=brain_mask,
        csf_mask=csf_mask,
        wm_mask=wm_mask,
    )


@pytest.mark.slow
def test_nuisance_36param(test_subject: TestSubjectData) -> None:
    """Test 36-parameter nuisance regression on short BOLD."""
    data = _prepare_bold_and_masks(test_subject)

    result = nuisance_regression(
        bold_file=data.bold,
        brain_mask_file=data.brain_mask,
        csf_mask_file=data.csf_mask,
        wm_mask_file=data.wm_mask,
        motion_par_file=data.par,
        regressor_set="36-parameter",
        bandpass=None,
    )

    assert result.cleaned_bold.exists()
    assert nifti_num_volumes(result.cleaned_bold) == 50
    assert result.regressor_file.exists()
    assert len(result.column_names) == 36
    assert result.eroded_masks.csf.any()
    assert result.eroded_masks.wm.any()


@pytest.mark.slow
def test_nuisance_acompcor(test_subject: TestSubjectData) -> None:
    """Test aCompCor nuisance regression on short BOLD."""
    data = _prepare_bold_and_masks(test_subject)

    result = nuisance_regression(
        bold_file=data.bold,
        brain_mask_file=data.brain_mask,
        csf_mask_file=data.csf_mask,
        wm_mask_file=data.wm_mask,
        motion_par_file=data.par,
        regressor_set="aCompCor",
        bandpass=None,
    )

    assert result.cleaned_bold.exists()
    assert nifti_num_volumes(result.cleaned_bold) == 50
    assert result.regressor_file.exists()
    assert len(result.column_names) == 37
    assert result.eroded_masks.csf.any()
    assert result.eroded_masks.wm.any()
