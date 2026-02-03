"""RBC functional initialization."""

from pathlib import Path

from niwrap import afni


def truncate_trs(
    in_file: Path, output_prefix: str, start_tr: int
) -> afni.V3dcalcOutputs:
    """Remove first N TRs from BOLD timeseries using AFNI 3dcalc.

    Args:
        in_file: Path to input BOLD timeseries to be truncated.
        output_prefix: Prefix of output file.
        start_tr: Number of TRs to remove from beginning.
    """
    return afni.v_3dcalc(
        in_file_a=in_file,
        expr="a",
        start_idx=start_tr,
        prefix=output_prefix,
    )

def scale(
    in_file: Path, scale_factor: float = 0.1
) -> afni.V3drefitOutputs:
    """Scale BOLD voxel dimensions using AFNI 3drefit.

    Args:
        in_file: Path to input BOLD timeseries to be scaled.
        scale_factor: Factor to scale voxel dimensions (default: 0.1 to divide by 10).
    """
    return afni.v_3drefit(
        in_file=in_file,
        xyzscale=scale_factor,
    )