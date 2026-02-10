"""RBC functional initialization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

if TYPE_CHECKING:
    from pathlib import Path


def truncate_trs(in_file: Path, start_tr: int) -> afni.V3dcalcOutputs:
    """Remove first N TRs from BOLD timeseries using AFNI 3dcalc.

    Args:
        in_file: Path to input BOLD timeseries to be truncated.
        start_tr: Number of TRs to remove from beginning.

    Returns:
        AFNI 3dcalc output object.
    """
    return afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=in_file, selectors_=f"[{start_tr}..$]"
        ),
        expression="a",
        prefix="truncated.nii.gz",
    )


def scale_bold(in_file: Path, scale_factor: float = 0.1) -> afni.V3drefitOutputs:
    """Scale BOLD voxel dimensions using AFNI 3drefit.

    Args:
        in_file: Path to input BOLD timeseries to be scaled.
        scale_factor: Factor to scale voxel dimensions (default: 0.1 to divide by 10).

    Returns:
        AFNI 3drefit output object.
    """
    return afni.v_3drefit(
        in_file=in_file,
        xyzscale=scale_factor,
    )
