"""RBC motion reference & correction."""

from pathlib import Path

from types import SimpleNamespace

from niwrap import afni
from niwrap import fsl


def generate_motion_reference(
    in_file: Path, output_fname: str
) -> afni.V3dcalcOutputs:
    """Creates reference volume for motion correction by extracting middle volume.

    Args:
        in_file: Path to input BOLD timeseries.
        output_fname: Name of output file.
    """

    total_vols = afni.v_3dinfo(
        dataset=[in_file], 
        nv=True
    )
    
    mid_vol = (int(total_vols.info[0])) // 2

    return afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=in_file,
            selectors_=f"[{mid_vol}]"
        ),
        expression="a",
        prefix=output_fname,
    )



