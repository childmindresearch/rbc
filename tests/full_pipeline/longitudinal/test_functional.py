"""Full e2e test for the longitudinal functional preprocessing workflow.

Tier-4 test for Stage 5 of the longitudinal refactor (tracker: #301).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np

if TYPE_CHECKING:
    from full_pipeline.longitudinal.conftest import LongitudinalPipelineData


def test_longitudinal_func_outputs_exist(
    longitudinal_pipeline_data: LongitudinalPipelineData,
) -> None:
    """All FunctionalLongOutputs paths exist on disk."""
    outputs = longitudinal_pipeline_data.long_func

    for field_name, value in outputs._asdict().items():
        if isinstance(value, dict):
            for key, path in value.items():
                assert Path(path).exists(), (
                    f"{field_name}[{key!r}] does not exist: {path}"
                )
        elif value is not None:
            assert Path(value).exists(), f"{field_name} does not exist: {value}"


def test_regressed_bold_non_degenerate(
    longitudinal_pipeline_data: LongitudinalPipelineData,
) -> None:
    """Regressed BOLD in longitudinal space has non-zero variance."""
    outputs = longitudinal_pipeline_data.long_func
    for reg, path in outputs.regressed_bold.items():
        img = nib.nifti1.load(path)
        data = img.get_fdata()
        assert data.var() > 0, f"Regressed BOLD for {reg!r} has zero variance"


def test_cleaned_bold_non_degenerate(
    longitudinal_pipeline_data: LongitudinalPipelineData,
) -> None:
    """Cleaned (bandpassed) BOLD in longitudinal space has non-zero variance."""
    outputs = longitudinal_pipeline_data.long_func
    for reg, path in outputs.cleaned_bold.items():
        img = nib.nifti1.load(path)
        data = img.get_fdata()
        assert data.var() > 0, f"Cleaned BOLD for {reg!r} has zero variance"


def test_bold_mask_is_binary(
    longitudinal_pipeline_data: LongitudinalPipelineData,
) -> None:
    """Warped bold mask in longitudinal space is binary."""
    mask_path = longitudinal_pipeline_data.long_func.bold_mask
    img = nib.nifti1.load(mask_path)
    data = img.get_fdata()
    unique = np.unique(data)
    assert set(unique).issubset({0, 1}), f"Mask has non-binary values: {unique}"
