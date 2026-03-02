"""Transform subject data to a longitudinal tempalte space."""

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

    res = ants.ants_apply_transforms(
        reference_image=template,
        input_image=in_file,
        output=ants.ants_apply_transforms_warped_output("subject_to_template.nii.gz"),
        dimensionality=3,
        interpolation=ants.ants_apply_transforms_linear(),
    )

    return res.output.output_image_outfile
