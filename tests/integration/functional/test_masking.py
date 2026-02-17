"""Tests for BOLD masking (FSL-AFNI method)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional.masking import bold_masking
from rbc.core.functional.motion import extract_motion_reference
from rbc.core.resources import MNI_TEMPLATES

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_bold_masking_outputs_exist(test_subject: TestSubjectData) -> None:
    """Test bold_masking produces both output files."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    bold_ref = extract_motion_reference(in_file=reoriented.out_file)
    result = bold_masking(
        bold_ref=bold_ref.output_file,
        template_mask=MNI_TEMPLATES.brain_mask_2mm,
        template_ref=MNI_TEMPLATES.bold_ref,
    )

    assert result.final_mask.exists(), "final_mask does not exist"
    assert result.skull_stripped_bold.exists(), "skull_stripped_bold does not exist"


@pytest.mark.slow
def test_bold_masking_final_mask_is_binary(test_subject: TestSubjectData) -> None:
    """Test final mask contains only 0s and 1s."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    bold_ref = extract_motion_reference(in_file=reoriented.out_file)

    result = bold_masking(
        bold_ref=bold_ref.output_file,
        template_mask=MNI_TEMPLATES.brain_mask_2mm,
        template_ref=MNI_TEMPLATES.bold_ref,
    )

    mask_img = nib.load(result.final_mask)
    mask_data = mask_img.get_fdata()
    unique_vals = np.unique(np.round(mask_data))
    assert set(unique_vals).issubset({0, 1}), (
        f"Mask has non-binary values: {unique_vals}"
    )
    assert 1 in unique_vals, "Mask is empty (all zeros)!"
