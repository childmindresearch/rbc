"""Single-step resampling to template space.

Applies motion correction, coregistration, and normalization transforms to
all volumes of a slice-timing corrected BOLD timeseries before merging back
into a 4D timeseries in template space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

import nibabel as nib
from niwrap import ants, fsl

from rbc.core.common import mat_to_itk


def split_4d(img_4d: Path) -> list[Path]:
    """Split a 4D NIfTI timeseries into individual 3D volumes.

    Args:
        img_4d: Path to a 4D NIfTI image.

    Returns:
        Sorted list of paths to the individual 3D volume files.
    """
    split_result = fsl.fslsplit(
        infile=img_4d, separation_time=True, output_basename="vol_"
    )
    assert split_result.out_files is not None  # noqa: S101
    out_files = split_result.out_files
    out_dir = out_files[0].parent if isinstance(out_files, list) else out_files.parent
    return sorted(out_dir.glob("vol_*.nii.gz"))


def merge_3d_to_4d(volumes: Sequence[Path], output: Path) -> Path:
    """Merge a sequence of 3D NIfTI volumes into a single 4D timeseries.

    Args:
        volumes: Ordered sequence of paths to 3D NIfTI images.
        output: Path to write the merged 4D image.

    Returns:
        Path to the merged 4D NIfTI image.
    """
    imgs = [nib.nifti1.load(v) for v in volumes]
    merged = nib.funcs.concat_images(imgs, axis=None)
    nib.save(merged, output)
    return output


def resample_bold_to_template(
    stc_img: Path,
    motion_mat_dir: Path,
    bold_to_anat: Path,
    anat_to_template: Path,
    bold_ref: Path,
    template: Path,
    t1w_brain: Path,
    distortion_warp: Path | None = None,
) -> Path:
    """Resample a slice-timing corrected BOLD timeseries to template space.

    Args:
        stc_img: Slice-timing corrected 4D BOLD timeseries.
        motion_mat_dir: Directory of MAT_* motion matrices from mcflirt.
        bold_to_anat: BOLD to T1w affine (output from BBR).
        anat_to_template: T1w to template composite warp.
        bold_ref: Skull-stripped BOLD reference.
        template: Brain template in target space.
        t1w_brain: Skull-stripped T1w brain.
        distortion_warp: Optional distortion correction warp.

    Returns:
        Resampled 4D BOLD in template space.

    Raises:
        FileNotFoundError: No motion .mat files are found in the specified directory.
        ValueError: Number of motion matrix files found does not match the number
            of slice-timing corrected volumes.
    """
    motion_mats = sorted(motion_mat_dir.glob("MAT_*"))
    if not motion_mats:
        raise FileNotFoundError(f"No motion .mat files found in {motion_mat_dir}")

    bold2anat_itk = mat_to_itk(bold_to_anat, t1w_brain, bold_ref, "bold2anat.txt")

    # Split slice time corrected 4D into volumes
    stc_vols = split_4d(stc_img)

    if len(motion_mats) != len(stc_vols):
        raise ValueError(
            f"Count mismatch: ({len(motion_mats)}) mats, ({len(stc_vols)}) volumes"
        )

    base_transforms = [anat_to_template, bold2anat_itk]
    if distortion_warp:
        base_transforms.append(distortion_warp)

    # Convert motion .mat to ITK & apply all transforms per volume
    # Order: motion -> distortion -> bold2anat -> anat2template
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
                f"vol_{idx:04d}_transform.nii.gz"
            ),
        )
        transformed_vols.append(result.output.output_image_outfile)

    # Merge transformed volumes back to 4D timeseries
    out_path = transformed_vols[0].parent / "bold_to_template_resampled.nii.gz"
    return merge_3d_to_4d(transformed_vols, out_path)
