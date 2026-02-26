"""BOLD initialization steps.

After reorientation (handled in ``rbc.core.common``), BOLD data undergoes
two initialization steps before motion correction:

1. **TR truncation** -- discard the first *N* volumes (default 2) to allow
   the scanner signal to reach steady state.
2. **Voxel scaling** -- rescale voxel dimensions (divide by 10) to match
   the coordinate conventions expected downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

if TYPE_CHECKING:
    from pathlib import Path


def truncate_trs(in_file: Path, start_tr: int) -> Path:
    """Discard the first *N* TRs from a BOLD timeseries.

    Early volumes are typically discarded because the MR signal has not yet
    reached a steady state, which would introduce intensity artifacts.

    Args:
        in_file: Reoriented BOLD timeseries.
        start_tr: Number of initial TRs to drop (e.g. 2).

    Returns:
        Path to the truncated BOLD timeseries.
    """
    result = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=in_file, selectors_=f"[{start_tr}..$]"
        ),
        expression="a",
        prefix="truncated.nii.gz",
    )
    assert result.output_file is not None  # noqa: S101
    return result.output_file


def scale_bold(in_file: Path, scale_factor: float = 0.1) -> afni.V3drefitOutputs:
    """Rescale BOLD voxel dimensions via ``3drefit -xyzscale``.

    Some pipelines store BOLD data with inflated voxel sizes. This step
    multiplies all voxel dimensions by *scale_factor* (default 0.1, i.e.
    divide by 10) to bring them into the expected coordinate range.

    Note:
        This modifies the NIfTI header in-place; the voxel data are unchanged.

    Args:
        in_file: BOLD timeseries whose header should be updated.
        scale_factor: Multiplier for voxel dimensions (default 0.1).

    Returns:
        AFNI 3drefit outputs.
    """
    return afni.v_3drefit(
        in_file=in_file,
        xyzscale=scale_factor,
    )
