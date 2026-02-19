"""Integration tests for AFNI methods used across modalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rbc.core import anatomical

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_brain_extraction(test_subject: TestSubjectData) -> None:
    """Test brain extraction."""
    ants_bet_output = anatomical.ants_brain_extraction(in_file=test_subject.t1w)
    # Test extracted brain image exists
    assert ants_bet_output.brain_extracted_image is not None
    assert ants_bet_output.brain_extracted_image.exists()
    # Test brain mask exists
    assert ants_bet_output.brain_mask is not None
    assert ants_bet_output.brain_mask.exists()


@pytest.mark.slow
def test_tissue_segmentation(test_subject: TestSubjectData) -> None:
    """Test tissue segmentation."""
    segmentation = anatomical.fsl_segmentation(in_file=test_subject.t1w)
    tissue_mask = anatomical.fsl_tissue_masks(fast_result=segmentation)
    assert tissue_mask.csf.exists()
    assert tissue_mask.gm.exists()
    assert tissue_mask.wm.exists()


@pytest.mark.slow
def test_registration(test_subject: TestSubjectData) -> None:
    """Test anatomical registration."""
    composite_xfms = anatomical.ants_registration(in_file=test_subject.t1w)
    assert composite_xfms.forward.exists()
    assert composite_xfms.inverse.exists()
