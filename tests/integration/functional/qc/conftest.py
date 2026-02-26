"""Shared fixtures for functional QC integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import nibabel as nib
import numpy as np
import pytest
from niwrap import afni

from rbc.core.common import deoblique_and_reorient
from rbc.core.functional import (
    extract_motion_reference,
    fsl_motion_correction,
)

if TYPE_CHECKING:
    from conftest import TestSubjectData

    from rbc.core.functional.motion import MotionCorrectedOutputs


class MotionCorrectedBOLD(NamedTuple):
    """Preprocessed BOLD data for QC tests."""

    mc: MotionCorrectedOutputs
    bold_data: np.ndarray
    mask: np.ndarray


@pytest.fixture(scope="session")
def motion_corrected_bold(test_subject: TestSubjectData) -> MotionCorrectedBOLD:
    """Deoblique, truncate to 10 volumes, and motion-correct test BOLD data."""
    reoriented = deoblique_and_reorient(in_file=test_subject.bold)
    truncated = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=reoriented.out_file, selectors_="[0..9]"
        ),
        expression="a",
        prefix="qc_shared_10vols.nii.gz",
    )

    assert truncated.output_file is not None
    ref = extract_motion_reference(in_file=truncated.output_file)
    mc = fsl_motion_correction(
        in_file=truncated.output_file,
        ref_file=ref,
    )

    bold_img = nib.nifti1.load(mc.bold)
    bold_data = bold_img.get_fdata()
    mask = np.mean(bold_data, axis=3) > 0

    return MotionCorrectedBOLD(mc=mc, bold_data=bold_data, mask=mask)
