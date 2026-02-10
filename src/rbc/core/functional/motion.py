"""RBC motion reference & correction."""

from pathlib import Path
from typing import NamedTuple, cast

import nibabel as nib
from niwrap import afni, fsl


def generate_motion_reference(in_file: Path, output_fname: str) -> afni.V3dcalcOutputs:
    """Creates reference volume for motion correction by extracting middle volume.

    Args:
        in_file: Path to input BOLD timeseries.
        output_fname: Name of output file.

    Returns:
        AFNI 3dcalc output object.
    """
    img = cast("nib.Nifti1Image", nib.load(in_file))
    total_vols = img.shape[3]

    mid_vol = total_vols // 2

    return afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(file=in_file, selectors_=f"[{mid_vol}]"),
        expression="a",
        prefix=output_fname,
    )


class MotionCorrectedOutputs(NamedTuple):
    """NamedTuple for motion correction outputs."""

    bold: Path
    par: Path
    rms_rel: Path
    rms_abs: Path
    mat_dir: Path


def motion_correction(
    in_file: Path, ref_file: Path, output_prefix: str
) -> MotionCorrectedOutputs:
    """Estimate and correct head motion using FSL mcflirt.

    Args:
        in_file: Path to input BOLD timeseries to correct.
        ref_file: Path to reference volume for motion correction.
        output_prefix: Prefix for output files.

    Returns:
        NamedTuple with paths to motion corrected outputs and matrices.
    """
    mc_result = fsl.mcflirt(
        in_file=in_file,
        ref_file=ref_file,
        save_mats=True,
        save_plots=True,
        save_rmsrel=True,
        save_rmsabs=True,
        out_file=output_prefix,
    )

    motion_mat_dir = Path(mc_result.root) / f"{output_prefix}.mat"

    if not motion_mat_dir.exists():
        raise FileNotFoundError(f"Missing .mat directory at {motion_mat_dir}")

    return MotionCorrectedOutputs(
        bold=Path(mc_result.out_file),
        par=Path(mc_result.par_file),
        rms_rel=Path(mc_result.rmsrel_files),
        rms_abs=Path(mc_result.rmsabs_files),
        mat_dir=motion_mat_dir,
    )
