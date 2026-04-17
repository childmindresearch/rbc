"""Integration tests for spatial smoothing of derivative maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import pytest

from rbc.core.common import deoblique_and_reorient, smooth

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_smooth_runs(test_subject: TestSubjectData) -> None:
    """Test that smooth runs successfully and produces output."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    result = smooth(in_file=reoriented.out_file, mask_file=reoriented.out_file)
    assert result.exists()


@pytest.mark.slow
def test_smooth_preserves_shape(test_subject: TestSubjectData) -> None:
    """Test that smoothed output has the same shape as input."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    in_img = nib.nifti1.load(reoriented.out_file)

    result = smooth(in_file=reoriented.out_file, mask_file=reoriented.out_file)

    out_img = nib.nifti1.load(result)
    assert out_img.shape == in_img.shape
