"""Tests for BOLD timeseries slice time correction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import slice_timing_correction
from rbc.core.nifti import nifti_num_slices

if TYPE_CHECKING:
    from conftest import TestSubjectData


def test_slice_timing_correction(test_subject: TestSubjectData) -> None:
    """Test that slice timing correction runs successfully and produces output."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    corrected = slice_timing_correction(in_file=reoriented.out_file, tr=2)
    assert corrected.out_file.exists()


def test_slice_timing_tpattern_list(test_subject: TestSubjectData) -> None:
    """Test that tpattern accepts a list of slice timing offsets."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    num_slices = nifti_num_slices(reoriented.out_file)

    expected_1d = reoriented.out_file.parent / "SliceTiming.1D"

    # sample tpattern based on total slices
    tpattern = [0.1 * i for i in range(num_slices)]
    corrected = slice_timing_correction(
        in_file=reoriented.out_file, tr=2, tpattern=tpattern
    )

    assert expected_1d.exists()
    assert len(expected_1d.read_text().splitlines()) == num_slices
    assert corrected.out_file.exists()


def test_slice_timing_tpattern_string(test_subject: TestSubjectData) -> None:
    """Test that tpattern accepts a string acquisition pattern."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    corrected = slice_timing_correction(
        in_file=reoriented.out_file, tr=2.0, tpattern="alt+z"
    )
    assert corrected.out_file.exists()
