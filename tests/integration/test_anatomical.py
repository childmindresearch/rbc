"""Integration tests for AFNI methods used across modalities."""

from types import SimpleNamespace

import pytest

from rbc.core import anatomical


@pytest.mark.slow
def test_brain_extraction(test_subject: SimpleNamespace) -> None:
    """Test brain extraction."""
    ants_bet_output = anatomical.ants_brain_extraction(
        in_file=test_subject.t1w, output_prefix="test"
    )
    # Test extracted brain image exists
    assert (
        ants_bet_output.brain_extracted_image is not None
        and ants_bet_output.brain_extracted_image.exists()
    )
    # Test brain mask exists
    assert (
        ants_bet_output.brain_mask is not None and ants_bet_output.brain_mask.exists()
    )


@pytest.mark.slow
def test_tissue_segmentation(test_subject: SimpleNamespace) -> None:
    """Test tissue segmentation."""
    tissue_mask = anatomical.fsl_tissue_segmentation(
        in_file=test_subject.t1w, output_prefix="test"
    )
    assert (
        tissue_mask.csf.exists() and tissue_mask.gm.exists() and tissue_mask.wm.exists()
    )


@pytest.mark.slow
def test_registration(test_subject: SimpleNamespace) -> None:
    """Test anatomical registration."""
    composite_xfms = anatomical.ants_registration(
        in_file=test_subject.t1w, output_prefix="test"
    )
    assert composite_xfms.forward.exists() and composite_xfms.inverse.exists()
