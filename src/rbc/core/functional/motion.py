"""RBC motion reference & correction."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from niwrap import afni, fsl

from rbc.core.nifti import nifti_num_volumes

_MC_PREFIX = "mc"


def extract_motion_reference(in_file: Path) -> afni.V3dcalcOutputs:
    """Extract reference volume for motion correction from the middle of the timeseries.

    Args:
        in_file: Path to input BOLD timeseries.

    Returns:
        AFNI 3dcalc output object.
    """
    total_vols = nifti_num_volumes(in_file)
    mid_vol = total_vols // 2

    return afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(file=in_file, selectors_=f"[{mid_vol}]"),
        expression="a",
        prefix="motion_ref.nii.gz",
    )


class MotionCorrectedOutputs(NamedTuple):
    """NamedTuple for motion correction outputs."""

    bold: Path
    par: Path
    rms_rel: Path
    rms_abs: Path
    mat_dir: Path


def fsl_motion_correction(in_file: Path, ref_file: Path) -> MotionCorrectedOutputs:
    """Estimate and correct head motion using FSL mcflirt.

    Args:
        in_file: Path to input BOLD timeseries to correct.
        ref_file: Path to reference volume for motion correction.

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
        out_file=_MC_PREFIX,
    )

    motion_mat_dir = Path(mc_result.root) / f"{_MC_PREFIX}.mat"

    if not motion_mat_dir.exists():
        raise FileNotFoundError(f"Missing .mat directory at {motion_mat_dir}")

    return MotionCorrectedOutputs(
        bold=Path(mc_result.out_file),
        par=Path(mc_result.par_file),
        rms_rel=Path(mc_result.rmsrel_files),
        rms_abs=Path(mc_result.rmsabs_files),
        mat_dir=motion_mat_dir,
    )
