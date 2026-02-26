"""Integration tests for truncating TRs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import truncate_trs
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
    assert truncated_bold.exists()
    assert nifti_num_volumes(truncated_bold) == original_count - start_tr


def test_truncate_to_min_volume(test_subject: TestSubjectData) -> None:
    """Test truncating to minimum volume count of 1."""
    original_count = nifti_num_volumes(test_subject.bold)

    start_tr = original_count - 1
    truncated_bold = truncate_trs(
        in_file=test_subject.bold,
        start_tr=start_tr,
    )
    # Test truncated BOLD file volume count is 1
    assert nifti_num_volumes(truncated_bold) == 1
