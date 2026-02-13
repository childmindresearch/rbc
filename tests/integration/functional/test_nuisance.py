"""Integration tests for nuisance regression."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from niwrap import afni

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
    nuisance_regression,
)
from rbc.core.nifti import nifti_num_volumes

if TYPE_CHECKING:
    from conftest import TestSubjectData

CPAC_ANAT_DIR = (
    Path(__file__).parents[2]
    / "data"
    / "cpac_outputs"
    / "ds000001"
    / "output"
    / "pipeline_RBCv0"
    / "sub-01"
    / "ses-1"
    / "anat"
)


def _prepare_bold_and_motion(
    test_subject: TestSubjectData,
    n_vols: int = 10,
) -> tuple[Path, Path, Path]:
    """Prepare a short BOLD series and motion parameters for testing.

    Returns (bold_file, motion_par_file, brain_mask_file) paths.
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
    mc = fsl_motion_correction(
        in_file=truncated.output_file, ref_file=motion_ref.output_file
    )
    return (
        mc.bold.with_suffix(".nii.gz"),
        mc.par,
        CPAC_ANAT_DIR / "sub-01_ses-1_desc-brain_mask.nii.gz",
    )


@pytest.mark.slow
def test_nuisance_36param(test_subject: TestSubjectData) -> None:
    """Test 36-parameter nuisance regression on short BOLD."""
    bold_file, par_file, brain_mask = _prepare_bold_and_motion(test_subject)

    csf_mask = CPAC_ANAT_DIR / "sub-01_ses-1_label-CSF_mask.nii.gz"
    wm_mask = CPAC_ANAT_DIR / "sub-01_ses-1_label-WM_mask.nii.gz"

    result = nuisance_regression(
        bold_file=bold_file,
        brain_mask_file=brain_mask,
        csf_mask_file=csf_mask,
        wm_mask_file=wm_mask,
        motion_par_file=par_file,
        regressor_set="36-parameter",
    )

    assert result.cleaned_bold.exists()
    assert nifti_num_volumes(result.cleaned_bold) == 10
    assert result.regressor_file.exists()
    assert len(result.column_names) == 36
    assert result.eroded_masks.csf.any()
    assert result.eroded_masks.wm.any()


@pytest.mark.slow
def test_nuisance_acompcor(test_subject: TestSubjectData) -> None:
    """Test aCompCor nuisance regression on short BOLD."""
    bold_file, par_file, brain_mask = _prepare_bold_and_motion(test_subject)

    csf_mask = CPAC_ANAT_DIR / "sub-01_ses-1_label-CSF_mask.nii.gz"
    wm_mask = CPAC_ANAT_DIR / "sub-01_ses-1_label-WM_mask.nii.gz"

    result = nuisance_regression(
        bold_file=bold_file,
        brain_mask_file=brain_mask,
        csf_mask_file=csf_mask,
        wm_mask_file=wm_mask,
        motion_par_file=par_file,
        regressor_set="aCompCor",
    )

    assert result.cleaned_bold.exists()
    assert nifti_num_volumes(result.cleaned_bold) == 10
    assert result.regressor_file.exists()
    assert len(result.column_names) == 37
    assert result.eroded_masks.csf.any()
    assert result.eroded_masks.wm.any()
