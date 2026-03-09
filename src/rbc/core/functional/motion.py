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
_MAX_VOLUMES = 50
_MIDDLE_SLICE_START = 20
_MIDDLE_SLICE_END = 40


class MotionCorrectedOutputs(NamedTuple):
    """Outputs from FSL mcflirt motion correction.

    Attributes:
        bold: Motion-corrected BOLD timeseries.
        motion_params: Normalized six-column motion parameter file
            (AFNI convention: [roll, pitch, yaw, dS, dL, dP],
            rotations in degrees).
        rms_rel: Frame-to-frame (relative) RMS displacement.
        rms_abs: Volume-to-reference (absolute) RMS displacement.
        mat_dir: Directory containing per-volume affine matrices.
    """

    bold: Path
    motion_params: Path
    rms_rel: Path
    rms_abs: Path
    mat_dir: Path


def extract_motion_reference(in_file: Path) -> Path:
    """Extract a motion-corrected reference image from BOLD timeseries.

    This follows the fMRIPrep approach of:
    1. Extracting up to 50 volumes from the input file.
    2. Selecting middle 20 volumes (volumes 20-40) if available.
    3. Applying motion correction using AFNI's ``3dvolreg``.
    4. Computing temporal median to create the final reference image.

    Args:
        in_file: BOLD timeseries.

    Returns:
        Motion reference image.
    """
    img = nib.squeeze_image(nib.load(in_file))

    if img.dataobj.ndim == 3:
        ref_volumes = [img]
    elif img.dataobj.ndim == 4:
        ref_volumes = nib.four_to_three(img.slicer[..., :_MAX_VOLUMES])
    else:
        raise ValueError(f"Unexpected number of dimensions: {img.dataobj.ndim}")

    ref_im = nib.squeeze_image(nib.concat_images(ref_volumes))
    # Clear header extensions to avoid shape-dependent inconsistencies after slicing
    ref_im.header.extensions.clear()

    # Middle volumes selection; fallback to all volumes if fewer than 40 are available
    if ref_im.ndim == 4 and ref_im.shape[-1] > _MIDDLE_SLICE_END:
        ref_im = nib.Nifti1Image(
            ref_im.dataobj[..., _MIDDLE_SLICE_START:_MIDDLE_SLICE_END],
            affine=ref_im.affine,
            header=ref_im.header,
        )

    temp_slice_file = generate_exec_folder(suffix="motion_ref_input") / "slice.nii.gz"
    ref_im.to_filename(temp_slice_file)

    mc_output_prefix = f"{_MC_PREFIX}_volreg.nii.gz"
    volreg_result = afni.v_3dvolreg(
        prefix=mc_output_prefix,
        in_file=temp_slice_file,
        fourier=True,
        twopass=True,
        zpad=4,
    )

    mc_output_file = volreg_result.out_file
    mc_data = nib.nifti1.load(mc_output_file).get_fdata()
    median_volume = np.median(mc_data, axis=3)

    output_file = (
        generate_exec_folder(suffix="motion_ref_output") / "motion_reference.nii.gz"
    )
    motion_ref_img = nib.Nifti1Image(
        median_volume, affine=ref_im.affine, header=ref_im.header
    )
    motion_ref_img.to_filename(output_file)

    return output_file


def normalize_motion_parameters(in_file: Path) -> Path:
    """Convert FSL mcflirt motion parameters to AFNI space.

    Converts rotations from radians to degrees and reorders/reorients
    axes from FSL to AFNI convention:
        FSL order: [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
        AFNI order: [roll, pitch, yaw, dS, dL, dP]

    Args:
        in_file: Path to mcflirt .par file (rotations in radians).

    Returns:
        Path to normalized motion_params.1D (rotations in degrees).
    """
    motion_params = np.genfromtxt(in_file).T  # (6, T)
    motion_params = np.vstack(
        (
            motion_params[2, :] * 180 / np.pi,  # roll  (FSL rot_z to degrees)
            motion_params[0, :] * 180 / np.pi,  # pitch (FSL rot_x to degrees)
            -motion_params[1, :] * 180 / np.pi,  # yaw   (FSL rot_y to degrees, flipped)
            motion_params[5, :],  # dS    (FSL trans_z)
            motion_params[3, :],  # dL    (FSL trans_x)
            -motion_params[4, :],  # dP    (FSL trans_y, flipped)
        )
    )
    motion_params = motion_params.T  # (T, 6)
    out_file = generate_exec_folder(suffix="motion_params") / "motion_params.1D"
    np.savetxt(out_file, motion_params)
    return out_file


def fsl_motion_correction(in_file: Path, ref_file: Path) -> MotionCorrectedOutputs:
    """Correct head motion using FSL ``mcflirt``.

    Each volume is rigidly aligned to the reference image. The motion parameters
    are normalized via :func:`normalize_motion_parameters` (FSL to AFNI convention)
    and are used downstream for nuisance regression and QC. The per-volume affine
    matrices are saved so they can later be composed with other transforms
    (distortion correction, coregistration, template warp) for single-step
    resampling to template space.

    Args:
        in_file: BOLD timeseries to motion-correct.
        ref_file: Single-volume reference image (from :func:`extract_motion_reference`).

    Returns:
        Motion-corrected data, normalized motion parameter file, displacement metrics,
        and the per-volume transform matrix directory.
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

    # niwrap returns the user-supplied prefix; FSL appends ".nii.gz" itself.
    bold_path = Path(f"{mc_result.out_file}.nii.gz")

    assert mc_result.par_file is not None  # noqa: S101
    assert mc_result.rmsrel_files is not None  # noqa: S101
    assert mc_result.rmsabs_files is not None  # noqa: S101

    motion_params = normalize_motion_parameters(mc_result.par_file)

    return MotionCorrectedOutputs(
        bold=bold_path,
        motion_params=motion_params,
        rms_rel=mc_result.rmsrel_files,
        rms_abs=mc_result.rmsabs_files,
        mat_dir=motion_mat_dir,
    )
