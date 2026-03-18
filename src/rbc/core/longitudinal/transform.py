"""Transform subject data to a longitudinal template space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import ants

if TYPE_CHECKING:
    from pathlib import Path


def anat_transform(in_file: Path, template: Path, xfm: Path) -> Path:
    """Apply existing transformation to anatomical data using ANTs.

    Args:
        in_file: Input file path to apply transform to.
        template: Longitudinal template file path for reference.
        xfm: Transformation to apply.

    Returns:
        Path to transformed file.

    Raises:
        FileNotFoundError: if input file or transformation not found.
    """
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")
    if not xfm.exists():
        raise FileNotFoundError(f"Transformation not found: {xfm}")

    return ants.ants_apply_transforms(
        reference_image=template,
        input_image=in_file,
        output=ants.ants_apply_transforms_warped_output("subject_to_template.nii.gz"),
        dimensionality=3,
        interpolation=ants.ants_apply_transforms_linear(),
    ).output.output_image_outfile


def compose_transform(ref: Path, bold_to_anat_xfm: Path, anat_to_tpl_xfm: Path) -> Path:
    """Compose single transformation from bold to template with ANTs.

    Args:
        ref: Reference image.
        bold_to_anat_xfm: Transformation from bold to anatomical space.
        anat_to_tpl_xfm: Transformation from anatomical to longitudinal template space.

    Returns:
        Path to composite transformation from bold to template space.

    Raises:
        FileNotFoundError: if a transformation is not found.
    """
    for fpath in (ref, bold_to_anat_xfm, anat_to_tpl_xfm):
        if not fpath.exists():
            raise FileNotFoundError(f"{fpath} not found")

    return ants.ants_apply_transforms(
        reference_image=ref,
        transform=[
            ants.ants_apply_transforms_transform_file_name(bold_to_anat_xfm),
            ants.ants_apply_transforms_transform_file_name(anat_to_tpl_xfm),
        ],
        output=ants.ants_apply_transforms_composite_displacement_field_output(
            composite_displacement_field="bold_to_tpl_xfm.nii.gz",
            print_out_composite_warp_file=True,
        ),
    ).output.output_image_outfile


def func_transform(in_file: Path, template: Path, xfm: Path) -> Path:
    """Apply transformation to functional data using ANTs.

    Args:
        in_file: Input file path to apply transform to.
        template: Longitudinal template file path for reference.
        xfm: Transformation to apply.

    Returns:
        Path to transformed file.

    Raises:
        FileNotFoundError: if input file or transformation not found.
    """
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")
    if not xfm.exists():
        raise FileNotFoundError(f"Transformation not found: {xfm}")

    return ants.ants_apply_transforms(
        dimensionality=4,
        reference_image=template,
        input_image=in_file,
        input_image_type=3,
        output=ants.ants_apply_transforms_warped_output("subj_bold_to_template.nii.gz"),
        interpolation=ants.ants_apply_transforms_linear(),
    ).output.output_image_outfile
