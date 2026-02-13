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

import nibabel as nib
import numpy as np
from niwrap import afni, fsl

from rbc.core.niwrap import generate_exec_folder

_MC_PREFIX = "mc"
MAX_VOLUMES = 50
_MIDDLE_SLICE_START = 20
_MIDDLE_SLICE_END = 40


class MotionReferenceOutputs(NamedTuple):
    """Outputs from motion reference extraction.

    Attributes:
        output_file: Path to the motion reference image.
    """

    output_file: Path


def extract_motion_reference(in_file: Path) -> MotionReferenceOutputs:
    """Extract a motion-corrected reference image from BOLD timeseries.

    This follows the fMRIPrep approach of:
    1. Extracting 50 volumes from the input file.
    2. Selecting middle 20 volumes (volumes 20-40) if more than 40 volumes are present.
    3. Apply motion correction using AFNI's ``3dvolreg``.
    4. Compute temporal median to create the reference image.

    Args:
        in_file: Truncated BOLD timeseries.

    Returns:
        Motion reference image.
    """
    img = nib.squeeze_image(nib.load(in_file))

    if img.dataobj.ndim == 3:
        ref_volumes = [img]
    elif img.dataobj.ndim == 4:
        ref_volumes = nib.four_to_three(img.slicer[..., :MAX_VOLUMES])
    else:
        raise ValueError(f"Unexpected number of dimensions: {img.dataobj.ndim}")

    ref_im = nib.squeeze_image(nib.concat_images(ref_volumes))
    ref_im.header.extensions.clear()

    if ref_im.shape[-1] > _MIDDLE_SLICE_END:
        ref_im = nib.Nifti1Image(
            ref_im.dataobj[..., _MIDDLE_SLICE_START:_MIDDLE_SLICE_END],
            affine=ref_im.affine,
            header=ref_im.header,
        )

    exec_dir = generate_exec_folder()
    temp_slice_file = exec_dir / "slice.nii.gz"
    ref_im.to_filename(temp_slice_file)

    mc_output_prefix = f"{_MC_PREFIX}_volreg.nii.gz"
    volreg_result = afni.v_3dvolreg(
        prefix=mc_output_prefix,
        in_file=str(temp_slice_file),
        fourier=True,
        twopass=True,
        zpad=4,
    )

    mc_output_file = Path(volreg_result.out_file)
    mc_data = nib.nifti1.load(mc_output_file).get_fdata()
    median_volume = np.median(mc_data, axis=3)

    output_file = exec_dir / "motion_reference.nii.gz"
    motion_ref_img = nib.Nifti1Image(
        median_volume, affine=ref_im.affine, header=ref_im.header
    )
    motion_ref_img.to_filename(output_file)

    return MotionReferenceOutputs(output_file=output_file)


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
