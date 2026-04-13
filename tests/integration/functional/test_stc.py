"""Tests for BOLD timeseries slice time correction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rbc.core.functional import slice_timing_correction
from rbc.core.nifti import nifti_num_slices

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_slice_timing_correction(test_subject: TestSubjectData) -> None:
    """Test that slice timing correction runs successfully and produces output."""
    corrected = slice_timing_correction(in_file=test_subject.bold, tr=2)
    assert corrected.exists()


@pytest.mark.slow
def test_slice_timing_tpattern_list(test_subject: TestSubjectData) -> None:
    """Test that tpattern accepts a list of slice timing offsets."""
    num_slices = nifti_num_slices(test_subject.bold)

    # sample tpattern based on total slices, kept within [0, tr)
    tpattern = [round(i * 2.0 / num_slices, 4) for i in range(num_slices)]
    corrected = slice_timing_correction(
        in_file=test_subject.bold, tr=2.0, tpattern=tpattern
    )
    assert corrected.exists()


@pytest.mark.slow
def test_slice_timing_tpattern_string(test_subject: TestSubjectData) -> None:
    """Test that tpattern accepts a string acquisition pattern."""
    corrected = slice_timing_correction(
        in_file=test_subject.bold, tr=2.0, tpattern="alt+z"
    )
    assert corrected.exists()
