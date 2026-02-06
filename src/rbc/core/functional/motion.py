"""RBC motion reference & correction."""

from pathlib import Path
from types import SimpleNamespace

from niwrap import afni, fsl


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
) -> SimpleNamespace:
    """Estimate and correct head motion using FSL mcflirt.

    Args:
        in_file: Path to input BOLD timeseries to correct.
        ref_file: Path to reference volume for motion correction.
        output_fname: Name of output file.
    """

    mc_result = fsl.mcflirt(
        in_file=in_file,
        ref_file=ref_file,
        save_mats=True,
        save_plots=True,
        save_rmsrel=True,
        save_rmsabs=True,
        out_file=f"{output_prefix}_mc_bold",
    )

    motion_mat_dir = [d for d in Path(mc_result.root).iterdir() if d.is_dir() and d.suffix == ".mat"]

    return SimpleNamespace(
        bold = Path(mc_result.out_file),
        par = Path(mc_result.par_file),
        rms_rel = Path(mc_result.rmsrel_files),
        rms_abs = Path(mc_result.rmsabs_files),
        mat_dir=motion_mat_dir[0]
    )