"""Resampling utilities for BOLD timeseries.

Provides two functions:

- :func:`apply_motion_transforms`: applies per-volume mcflirt affines to
  STC volumes, producing desc-preproc_bold (MC + STC in native space).
  This is an exported derivative only, not consumed by downstream steps.
- :func:`resample_bold_to_template`: single-step resampling of STC BOLD
  to template space, applying motion + BBR + anat-to-template transforms
  in one interpolation pass per volume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib

if TYPE_CHECKING:
    from pathlib import Path

from niwrap import ants

from rbc.core.common import mat_to_itk, merge_3d_to_4d, split_4d


def _restore_tr(resampled: Path, source: Path) -> None:
    """Copy pixdim[4] (TR) from *source* into *resampled* NIfTI header.

    antsApplyTransforms sets pixdim from the reference image, which for
    template-space outputs is a 3D template with no meaningful TR.  This
    restores the original temporal zoom from the source BOLD.
    """
    src_img = nib.nifti1.load(source)
    res_img = nib.nifti1.load(resampled, mmap=False)
    zooms = res_img.header.get_zooms()[:3] + src_img.header.get_zooms()[3:]
    res_img.header.set_zooms(zooms)
    nib.save(res_img, resampled)


def apply_motion_transforms(
    stc_img: Path,
    motion_mat_dir: Path,
    bold_ref: Path,
) -> Path:
    """Apply mcflirt motion affines to STC volumes (desc-preproc_bold).

    Produces a motion-corrected + slice-timing corrected BOLD in native
    space. Motion was estimated on pre-STC (despiked) data; here the
    resulting per-volume .mat affines are applied to the STC timeseries.

    The output is exported as desc-preproc_bold but is not consumed by
    any downstream workflow step. Template-space output uses
    :func:`resample_bold_to_template` instead (single interpolation pass).

    Args:
        stc_img: Slice-timing corrected 4D BOLD timeseries.
        motion_mat_dir: Directory of per-volume MAT_* matrices from mcflirt.
        bold_ref: BOLD reference volume (used as both source and reference
            for FSL-to-ITK matrix conversion).

    Returns:
        4D BOLD (MC + STC) in native space.

    Raises:
        FileNotFoundError: No motion .mat files are found in the specified directory.
        ValueError: Number of motion matrix files found does not match the number
            of slice-timing corrected volumes.
    """
    motion_mats = sorted(motion_mat_dir.glob("MAT_*"))
    if not motion_mats:
        raise FileNotFoundError(f"No motion .mat files found in {motion_mat_dir}")

    stc_vols = split_4d(stc_img)

    if len(motion_mats) != len(stc_vols):
        raise ValueError(
            f"Count mismatch: ({len(motion_mats)}) mats, ({len(stc_vols)}) volumes"
        )

    transformed_vols = []
    for idx, (motion_mat, stc_vol) in enumerate(
        zip(motion_mats, stc_vols, strict=True)
    ):
        motion_itk = mat_to_itk(motion_mat, bold_ref, bold_ref, f"motion_{idx:04d}.txt")
        result = ants.ants_apply_transforms(
            input_image=stc_vol,
            reference_image=bold_ref,  # bold in native space
            transform=[ants.ants_apply_transforms_transform_file_name(motion_itk)],
            interpolation=ants.ants_apply_transforms_lanczos_windowed_sinc(),
            float_=True,
            default_value=0,
            dimensionality=3,
            output=ants.ants_apply_transforms_warped_output(
                f"vol_{idx:04d}_motion.nii.gz"
            ),
        )
        transformed_vols.append(result.output.output_image_outfile)

    out_path = transformed_vols[0].parent / "preproc_bold.nii.gz"
    merged = merge_3d_to_4d(transformed_vols, out_path)

    # antsApplyTransforms writes the reference image's pixdim into the output
    # header, so pixdim[4] (TR) is lost.  Restore it from the source BOLD.
    _restore_tr(merged, stc_img)

    return merged


def resample_bold_to_template(
    stc_bold: Path,
    motion_mat_dir: Path,
    bold_to_anat: Path,
    anat_to_template: Path,
    bold_ref: Path,
    template: Path,
    t1w_brain: Path,
    distortion_warp: Path | None = None,
) -> Path:
    """Single-step resampling of STC BOLD to template space.

    Applies all spatial transforms (motion + BBR + anat-to-template, and
    optionally distortion correction) in a single ``antsApplyTransforms``
    call per volume. This avoids multiple interpolation passes.

    Args:
        stc_bold: Slice-timing corrected 4D BOLD timeseries.
        motion_mat_dir: Directory of per-volume MAT_* matrices from mcflirt.
        bold_to_anat: BOLD to T1w affine (output from BBR).
        anat_to_template: T1w to template composite warp.
        bold_ref: BOLD reference volume (used for ITK conversion).
        template: Brain template in target space.
        t1w_brain: Skull-stripped T1w brain.
        distortion_warp: Optional distortion correction warp.

    Returns:
        Resampled 4D BOLD in template space.

    Raises:
        FileNotFoundError: No motion .mat files found in the directory.
        ValueError: Number of motion matrices does not match STC volumes.
    """
    motion_mats = sorted(motion_mat_dir.glob("MAT_*"))
    if not motion_mats:
        raise FileNotFoundError(f"No motion .mat files found in {motion_mat_dir}")

    bold2anat_itk = mat_to_itk(bold_to_anat, t1w_brain, bold_ref, "bold2anat.txt")

    stc_vols = split_4d(stc_bold)

    if len(motion_mats) != len(stc_vols):
        raise ValueError(
            f"Count mismatch: ({len(motion_mats)}) mats, ({len(stc_vols)}) volumes"
        )

    # Shared transforms (applied to every volume)
    base_transforms = [anat_to_template, bold2anat_itk]
    if distortion_warp:
        base_transforms.append(distortion_warp)

    # Per-volume: all transforms in one antsApplyTransforms call
    # Order (last applied first): motion -> distortion -> bold2anat -> anat2template
    transformed_vols = []
    for idx, (motion_mat, stc_vol) in enumerate(
        zip(motion_mats, stc_vols, strict=True)
    ):
        motion_itk = mat_to_itk(motion_mat, bold_ref, bold_ref, f"motion_{idx:04d}.txt")
        transforms: list[
            ants.AntsApplyTransformsTransformFileNameParamsDictTagged
            | ants.AntsApplyTransformsUseInverseParamsDictTagged
        ] = [
            ants.ants_apply_transforms_transform_file_name(t)
            for t in [*base_transforms, motion_itk]
        ]
        result = ants.ants_apply_transforms(
            input_image=stc_vol,
            reference_image=template,
            transform=transforms,
            interpolation=ants.ants_apply_transforms_lanczos_windowed_sinc(),
            float_=True,
            default_value=0,
            dimensionality=3,
            output=ants.ants_apply_transforms_warped_output(
                f"vol_{idx:04d}_template.nii.gz"
            ),
        )
        transformed_vols.append(result.output.output_image_outfile)

    out_path = transformed_vols[0].parent / "bold_to_template_resampled.nii.gz"
    merged = merge_3d_to_4d(transformed_vols, out_path)

    # antsApplyTransforms writes the reference (template) image's pixdim into
    # the output header, so pixdim[4] (TR) is lost.  Restore it from the
    # source BOLD.
    _restore_tr(merged, stc_bold)

    return merged
