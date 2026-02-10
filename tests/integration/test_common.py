"""Integration tests for AFNI methods used across modalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

from rbc.core.common import reorient

if TYPE_CHECKING:
    from conftest import TestSubjectData


def test_reorient(test_subject: TestSubjectData) -> None:
    """Test deobliqueing and reorientation."""
    reoriented_file = reorient(test_subject.t1w, output_fname="test.nii.gz")
    assert (
        afni.v_3dinfo(dataset=[reoriented_file.out_file], orient=True).info[0] == "RPI"
    )
