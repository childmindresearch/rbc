"""Motion reference extraction and head-motion correction.

Before correcting motion, a single reference volume is extracted from the
middle of the BOLD timeseries. Every other volume is then realigned
to this reference using FSL ``mcflirt``, producing motion-corrected
data along with per-volume rigid-body parameters (3 rotations + 3 translations)
and displacement metrics used downstream for QC and nuisance regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from niwrap import afni, fsl

from rbc.core.nifti import nifti_num_volumes

_MC_PREFIX = "mc"


def extract_motion_reference(in_file: Path) -> afni.V3dcalcOutputs:
    """Extract the middle volume of a BOLD timeseries as a motion reference.

    The middle volume is chosen because it minimizes the maximum temporal
    distance to any other volume, reducing interpolation error during
    motion correction.

    Args:
        in_file: Truncated BOLD timeseries.

    Returns:
        AFNI 3dcalc outputs (use ``.output_file`` for the reference image).
    """
    total_vols = nifti_num_volumes(in_file)
    mid_vol = total_vols // 2

    return afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(file=in_file, selectors_=f"[{mid_vol}]"),
        expression="a",
        prefix="motion_ref.nii.gz",
    )


class MotionCorrectedOutputs(NamedTuple):
    """Outputs from FSL mcflirt motion correction.

    Attributes:
        bold: Motion-corrected BOLD timeseries.
        par: Six-column motion parameter file (3 rotations, 3 translations).
        rms_rel: Frame-to-frame (relative) RMS displacement.
        rms_abs: Volume-to-reference (absolute) RMS displacement.
        mat_dir: Directory containing per-volume affine matrices.
    """

    bold: Path
    par: Path
    rms_rel: Path
    rms_abs: Path
    mat_dir: Path


def fsl_motion_correction(in_file: Path, ref_file: Path) -> MotionCorrectedOutputs:
    """Estimate and correct head motion using FSL ``mcflirt``.

    Each volume is rigidly aligned to the reference image. The per-volume
    affine matrices are saved so they can later be composed with other
    transforms (distortion correction, coregistration, template warp) for
    single-step resampling to template space.

    Args:
        in_file: BOLD timeseries to motion-correct.
        ref_file: Single-volume reference image (from :func:`extract_motion_reference`).

    Returns:
        Motion-corrected data, parameter files, displacement metrics, and
        the per-volume transform matrix directory.
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
