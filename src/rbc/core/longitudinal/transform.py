"""Transform subject data to a longitudinal template space."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from niwrap import ants

from rbc.core.common import merge_3d_to_4d, split_4d
from rbc.core.functional.resampling import _restore_tr
from rbc.core.niwrap import generate_exec_folder

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
        transform=[ants.ants_apply_transforms_transform_file_name(xfm)],
        output=ants.ants_apply_transforms_warped_output("subject_to_template.nii.gz"),
        dimensionality=3,
        interpolation=ants.ants_apply_transforms_linear(),
    ).output.output_image_outfile


def compose_transform(ref: Path, bold_to_anat_itk: Path, anat_to_tpl_xfm: Path) -> Path:
    """Compose single transformation from bold to template with ANTs.

    Args:
        ref: Reference image.
        bold_to_anat_itk: Transformation from bold to anatomical space.
        anat_to_tpl_xfm: Transformation from anatomical to longitudinal template space.

    Returns:
        Path to composite transformation from bold to template space.

    Raises:
        FileNotFoundError: if a transformation is not found.
    """
    for fpath in (ref, bold_to_anat_itk, anat_to_tpl_xfm):
        if not fpath.exists():
            raise FileNotFoundError(f"{fpath} not found")

    return ants.ants_apply_transforms(
        reference_image=ref,
        transform=[
            ants.ants_apply_transforms_transform_file_name(anat_to_tpl_xfm),
            ants.ants_apply_transforms_transform_file_name(bold_to_anat_itk),
        ],
        output=ants.ants_apply_transforms_composite_displacement_field_output(
            composite_displacement_field="bold_to_tpl_xfm.nii.gz",
            print_out_composite_warp_file=True,
        ),
    ).output.output_image_outfile


def func_transform(
    in_file: Path,
    template: Path,
    xfm: Path,
    strategy: Literal["single", "chunked"] = "chunked",
) -> Path:
    """Apply transformation to functional data using ANTs.

    Args:
        in_file: Input file path to apply transform to.
        template: Longitudinal template file path for reference.
        xfm: Transformation to apply.
        strategy: Transformation strategy to apply.

    Returns:
        Path to transformed file.

    Raises:
        FileNotFoundError: if input file or transformation not found.
        ValueError: if strategy is unknown (not 'single' or 'chunked')
    """
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")
    if not xfm.exists():
        raise FileNotFoundError(f"Transformation not found: {xfm}")

    match strategy:
        case "chunked":
            return _transform_4d_chunked(in_file=in_file, template=template, xfm=xfm)
        case "single":
            return _transform_4d(in_file=in_file, template=template, xfm=xfm)
        case _:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. Must be 'single' or 'chunked'"
            )


def _transform_4d(in_file: Path, template: Path, xfm: Path) -> Path:
    """Apply transformation directly on 4D image."""
    return ants.ants_apply_transforms(
        dimensionality=3,
        reference_image=template,
        input_image=in_file,
        input_image_type=3,
        transform=[ants.ants_apply_transforms_transform_file_name(xfm)],
        output=ants.ants_apply_transforms_warped_output("subj_bold_to_template.nii.gz"),
        interpolation=ants.ants_apply_transforms_linear(),
    ).output.output_image_outfile


def _transform_4d_chunked(in_file: Path, template: Path, xfm: Path) -> Path:
    """Apply transformation using a chunked strategy (split 4D image into 3D)."""
    func_vols = split_4d(in_file)

    transformed_vols = []
    for idx, func_vol in enumerate(func_vols):
        result = ants.ants_apply_transforms(
            input_image=func_vol,
            reference_image=template,
            transform=[ants.ants_apply_transforms_transform_file_name(xfm)],
            float_=True,
            default_value=0,
            dimensionality=3,
            interpolation=ants.ants_apply_transforms_linear(),
            output=ants.ants_apply_transforms_warped_output(
                f"vol_{idx:04d}_template.nii"
            ),
        )
        transformed_vols.append(result.output.output_image_outfile)

    out_path = (
        generate_exec_folder("bold_to_longitudinal_merge")
        / "bold_to_longitudinal.nii.gz"
    )
    merged = merge_3d_to_4d(transformed_vols, out_path)
    _restore_tr(merged, in_file)
    return merged


def mask_transform(mask: Path, template: Path, xfm: Path) -> Path:
    """Apply transformation to mask using ANTs.

    Args:
        mask: Mask file path to apply transform to.
        template: Longitudinal template file path for reference.
        xfm: Transformation to apply.

    Returns:
        Path to transformed file.

    Raises:
        FileNotFoundError: if mask file or transformation not found.
    """
    if not mask.exists():
        raise FileNotFoundError(f"Mask file not found: {mask}")
    if not xfm.exists():
        raise FileNotFoundError(f"Transformation not found: {xfm}")

    return ants.ants_apply_transforms(
        input_image=mask,
        reference_image=template,
        transform=[ants.ants_apply_transforms_transform_file_name(xfm)],
        interpolation=ants.ants_apply_transforms_nearest_neighbor(),
        dimensionality=3,
        output=ants.ants_apply_transforms_warped_output("mask_to_template.nii.gz"),
    ).output.output_image_outfile
