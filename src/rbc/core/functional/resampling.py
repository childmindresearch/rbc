"""Single-step resampling to template space.

Applies motion correction, coregistration, and normalization transforms to
all volumes of a slice-timing corrected BOLD timeseries before merging back
into a 4D timeseries in template space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from niwrap import ants, c3d, fsl


def fsl_mat_to_itk(mat: Path, reference: Path, source: Path, output: str) -> Path:
    """Convert .mat affine to ITK .txt format using c3d_affine_tool."""
    result = c3d.c3d_affine_tool(
        reference_file=reference,
        source_file=source,
        transform_file=mat,
        out_itk_transform=output,
        fsl2ras=True,
    )
    return result.itk_transform_outfile


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
    """
    motion_mats = sorted(motion_mat_dir.glob("MAT_*"))
    if not motion_mats:
        raise FileNotFoundError(f"No motion .mat files found in {motion_mat_dir}")

    bold2anat_itk = fsl_mat_to_itk(bold_to_anat, t1w_brain, bold_ref, "bold2anat.txt")

    # Split slice time corrected 4D into volumes
    split_stc = fsl.fslsplit(
        infile=stc_img, separation_time=True, output_basename="vol_"
    )
    stc_vols = sorted(split_stc.out_files.parent.glob("vol_*.nii.gz"))

    base_transforms = [anat_to_template, bold2anat_itk]
    if distortion_warp:
        base_transforms.append(distortion_warp)

    # Convert motion .mat to ITK & apply all transforms per volume
    # Order: motion -> distortion -> bold2anat -> anat2template
    transformed_vols = []
    for idx in range(len(stc_vols)):
        motion_itk = fsl_mat_to_itk(
            motion_mats[idx], bold_ref, bold_ref, f"motion_{idx:04d}.txt"
        )
        transforms = [
            ants.ants_apply_transforms_transform_file_name(t)
            for t in [*base_transforms, motion_itk]
        ]
        result = ants.ants_apply_transforms(
            input_image=stc_vols[idx],
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
    return fsl.fslmerge(
        output_file="bold_to_template_resampled.nii.gz",
        input_files=transformed_vols,
        merge_time=True,
    )
