"""Integration tests for functional motion correction."""

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


def test_motion_reference_volume_count(test_subject: TestSubjectData) -> None:
    """Test motion reference volume count is 1."""
    reference = extract_motion_reference(in_file=test_subject.bold)
    # Test motion reference file has 1 volume
    assert nifti_num_volumes(reference) == 1


def test_motion_reference_short_series_fallback(test_subject: TestSubjectData) -> None:
    """Test fallback to all volumes when timeseries is < 40 volumes."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)

    # 5 volume dataset
    truncated_5 = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..4]"
        ),
        expression="a",
        prefix="test_5vols.nii.gz",
    )
    assert truncated_5.output_file is not None
    motion_reference = extract_motion_reference(in_file=truncated_5.output_file)
    assert nifti_num_volumes(motion_reference) == 1


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
    assert truncated_10.output_file is not None
    motion_reference = extract_motion_reference(in_file=truncated_10.output_file)
    motion_corrected = fsl_motion_correction(
        in_file=truncated_10.output_file,
        ref_file=motion_reference,
    )
    assert motion_corrected.bold.exists()
    assert nifti_num_volumes(motion_corrected.bold) == 10

    assert motion_corrected.motion_params.exists()
    par_data = motion_corrected.motion_params.read_text().splitlines()
    assert len(par_data) == 10


@pytest.mark.slow
def test_motion_correction(test_subject: TestSubjectData) -> None:
    """Test motion correction on full BOLD timeseries."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = truncate_trs(in_file=reoriented.out_file, start_tr=2)
    motion_reference = extract_motion_reference(in_file=truncated)
    motion_corrected = fsl_motion_correction(
        in_file=truncated,
        ref_file=motion_reference,
    )
    # Test motion corrected BOLD files exists
    assert motion_corrected.bold.exists()
    assert motion_corrected.motion_params.exists()
    assert motion_corrected.rms_rel.exists()
