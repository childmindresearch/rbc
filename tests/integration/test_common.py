"""Integration tests for AFNI methods used across modalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from niwrap import afni

from rbc.core.common import deoblique_and_reorient

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_deoblique_and_reorient(test_subject: TestSubjectData) -> None:
    """Test deobliqueing and reorientation."""
    reoriented_file = deoblique_and_reorient(test_subject.t1w)
    assert (
        afni.v_3dinfo(dataset=[reoriented_file.out_file], orient=True).info[0] == "RPI"
    )
