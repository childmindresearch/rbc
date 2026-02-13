"""Integration tests for nuisance regression."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

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


class _PreparedData(NamedTuple):
    bold: Path
    par: Path
    brain_mask: Path
    csf_mask: Path
    wm_mask: Path


def _resample_mask_to_bold(mask_file: Path, bold_file: Path, prefix: str) -> Path:
    """Resample an anat-space mask to the BOLD grid (NN interpolation)."""
    result = afni.v_3dresample(
        in_file=mask_file,
        prefix=prefix,
        master=bold_file,
        resample_mode="NN",
    )
    return result.out_file


def _prepare_bold_and_masks(
    test_subject: TestSubjectData,
    n_vols: int = 50,
) -> _PreparedData:
    """Prepare a short BOLD series, motion params, and resampled masks.

    The anat-space masks are resampled to the BOLD grid so that spatial
    dimensions match for nuisance regression.
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
    bold_file = mc.bold.with_suffix(".nii.gz")

    # Resample anat-space masks to BOLD grid
    brain_mask = _resample_mask_to_bold(
        CPAC_ANAT_DIR / "sub-01_ses-1_desc-brain_mask.nii.gz",
        bold_file,
        "brain_mask_bold.nii.gz",
    )
    csf_mask = _resample_mask_to_bold(
        CPAC_ANAT_DIR / "sub-01_ses-1_label-CSF_mask.nii.gz",
        bold_file,
        "csf_mask_bold.nii.gz",
    )
    wm_mask = _resample_mask_to_bold(
        CPAC_ANAT_DIR / "sub-01_ses-1_label-WM_mask.nii.gz",
        bold_file,
        "wm_mask_bold.nii.gz",
    )

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
