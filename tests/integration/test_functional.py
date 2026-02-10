"""Integration tests for functional workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from niwrap import afni

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
    truncate_trs,
)
from rbc.core.nifti import nifti_num_volumes

if TYPE_CHECKING:
    from conftest import TestSubjectData


def test_truncate_trs(test_subject: TestSubjectData) -> None:
    """Test truncating initial TRs from BOLD timeseries."""
    original_count = nifti_num_volumes(test_subject.bold)

    start_tr = 4
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated_bold = truncate_trs(
        in_file=reoriented.out_file,
        start_tr=start_tr,
    )
    # Test truncated BOLD file exists & volume count is reduced
    assert truncated_bold.output_file.exists()
    assert nifti_num_volumes(truncated_bold.output_file) == original_count - start_tr


def test_truncate_to_min_volume(test_subject: TestSubjectData) -> None:
    """Test truncating to minimum volume count of 1."""
    original_count = nifti_num_volumes(test_subject.bold)

    start_tr = original_count - 1
    truncated_bold = truncate_trs(
        in_file=test_subject.bold,
        start_tr=start_tr,
    )
    # Test truncated BOLD file volume count is 1
    assert nifti_num_volumes(truncated_bold.output_file) == 1


def test_motion_reference_volume_count(test_subject: TestSubjectData) -> None:
    """Test motion reference volume count is 1."""
    reference = extract_motion_reference(in_file=test_subject.bold)
    # Test motion reference file has 1 volume
    assert nifti_num_volumes(reference.output_file) == 1


@pytest.mark.slow
def test_motion_correction_10vols(test_subject: TestSubjectData) -> None:
    """Test motion correction on 10 volumes of BOLD timeseries."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated_10 = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..9]"
        ),
        expression="a",
        prefix="test_10vols.nii.gz",
    )
    motion_reference = extract_motion_reference(in_file=truncated_10.output_file)
    motion_corrected = fsl_motion_correction(
        in_file=truncated_10.output_file,
        ref_file=motion_reference.output_file,
    )
    assert motion_corrected.bold.with_suffix(".nii.gz").exists()
    assert nifti_num_volumes(motion_corrected.bold.with_suffix(".nii.gz")) == 10

    assert motion_corrected.par.exists()
    par_data = motion_corrected.par.read_text().splitlines()
    assert len(par_data) == 10


@pytest.mark.slow
def test_motion_correction(test_subject: TestSubjectData) -> None:
    """Test motion correction on full BOLD timeseries."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = truncate_trs(in_file=reoriented.out_file, start_tr=2)
    motion_reference = extract_motion_reference(in_file=truncated.output_file)
    motion_corrected = fsl_motion_correction(
        in_file=truncated.output_file,
        ref_file=motion_reference.output_file,
    )
    # Test motion corrected BOLD files exists
    assert motion_corrected.bold.with_suffix(".nii.gz").exists()
    assert motion_corrected.par.exists()
    assert motion_corrected.rms_rel.exists()
