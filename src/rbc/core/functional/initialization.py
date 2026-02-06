"""RBC functional initialization."""

from pathlib import Path

from niwrap import afni

def truncate_trs(
    in_file: Path, output_fname: str, start_tr: int 
) -> afni.V3dcalcOutputs:
    """Remove first N TRs from BOLD timeseries using AFNI 3dcalc.

    Args:
        in_file: Path to input BOLD timeseries to be truncated.
        output_fname: Name of output file.
        start_tr: Number of TRs to remove from beginning.

    Returns:
        AFNI 3dcalc output object.
    """
    return afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=in_file,
            selectors_=f"[{start_tr}..$]"
        ),
        expression="a",
        prefix=output_fname,
    )


def scale(
    in_file: Path, scale_factor: float = 0.1
) -> afni.V3drefitOutputs:
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