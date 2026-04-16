"""Integration test for longitudinal functional regression reuse.

Exercises the full ``rbc longitudinal functional`` CLI on ds000114,
which chains: composed BOLD-to-longitudinal warp, BOLD resampling,
and re-application of cross-sectional regressors in longitudinal space.

Tier-2 integration test for Stage 5 of the longitudinal refactor
(tracker: #301).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
def test_longitudinal_functional_produces_expected_tree(
    longitudinal_func_output: Path,
) -> None:
    """``rbc longitudinal functional`` writes per-regressor BOLD derivatives.

    Checks that the ``ses-test`` func directory under ``space-longitudinal``
    contains the expected files: warped BOLD, sbref, mask, composite xfm,
    and per-regressor regressed + cleaned BOLD.
    """
    func_dir = longitudinal_func_output / "sub-01" / "ses-test" / "func"
    stem = "sub-01_ses-test_task-fingerfootlips"

    expected_fragments = [
        f"{stem}_space-longitudinal_sbref.nii.gz",
        f"{stem}_space-longitudinal_desc-preproc_bold.nii.gz",
        f"{stem}_space-longitudinal_desc-brain_mask.nii.gz",
        f"{stem}_space-longitudinal_desc-regressed_reg-36parameter_bold.nii.gz",
        f"{stem}_space-longitudinal_desc-preproc_reg-36parameter_bold.nii.gz",
    ]
    tree = sorted(
        str(p.relative_to(longitudinal_func_output))
        for p in longitudinal_func_output.rglob("*")
        if p.is_file()
    )
    for name in expected_fragments:
        assert (func_dir / name).is_file(), (
            f"Missing: {name}\n--- file tree ---\n" + "\n".join(tree)
        )


@pytest.mark.slow
def test_regressed_bold_non_degenerate(
    longitudinal_func_output: Path,
) -> None:
    """Regressed BOLD in longitudinal space has non-zero variance."""
    path = (
        longitudinal_func_output
        / "sub-01"
        / "ses-test"
        / "func"
        / "sub-01_ses-test_task-fingerfootlips"
        "_space-longitudinal_desc-regressed_reg-36parameter_bold.nii.gz"
    )
    img = nib.nifti1.load(path)
    data = img.get_fdata()
    assert data.var() > 0, "Regressed BOLD has zero variance"


@pytest.mark.slow
def test_cleaned_bold_non_degenerate(
    longitudinal_func_output: Path,
) -> None:
    """Cleaned (bandpassed) BOLD in longitudinal space has non-zero variance."""
    path = (
        longitudinal_func_output
        / "sub-01"
        / "ses-test"
        / "func"
        / "sub-01_ses-test_task-fingerfootlips"
        "_space-longitudinal_desc-preproc_reg-36parameter_bold.nii.gz"
    )
    img = nib.nifti1.load(path)
    data = img.get_fdata()
    assert data.var() > 0, "Cleaned BOLD has zero variance"


@pytest.mark.slow
def test_bold_mask_is_binary(
    longitudinal_func_output: Path,
) -> None:
    """Warped bold mask in longitudinal space is binary."""
    path = (
        longitudinal_func_output
        / "sub-01"
        / "ses-test"
        / "func"
        / "sub-01_ses-test_task-fingerfootlips"
        "_space-longitudinal_desc-brain_mask.nii.gz"
    )
    img = nib.nifti1.load(path)
    data = img.get_fdata()
    unique = np.unique(data)
    assert set(unique).issubset({0, 1}), f"Mask has non-binary values: {unique}"
