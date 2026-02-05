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

def motion_correction(
        in_file: Path, ref_file: Path, output_prefix: str
) -> fsl.McflirtOutputs:
    """Estimate and correct head motion using FSL mcflirt.

    Args:
        in_file: Path to input BOLD timeseries to correct.
        ref_file: Path to reference volume for motion correction.
        output_prefix: Prefix of output file.
    """
    return fsl.mcflirt(
        in_file=in_file,
        ref_file=ref_file,
        save_mats=True,
        save_plots=True,
        save_rmsrel=True,
        save_rmsabs=True,
        out_file=output_prefix,
    )