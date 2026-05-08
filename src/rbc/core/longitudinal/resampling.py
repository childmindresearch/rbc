"""Resampling utiltiies for longitudinal templates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
from nibabel.processing import resample_from_to

if TYPE_CHECKING:
    from pathlib import Path

from rbc.core.niwrap import generate_exec_folder


def resample_template_to_bold(bold_ref: Path, template: Path) -> Path:
    """Resample template to BOLD grid if shapes differ.

    Args:
        bold_ref: BOLD reference volume (used for ITK conversion).
        template: Brain template in target space.

    Returns:
        Resampled template image with BOLD grid

    Raises:
        FileNotFoundError: No motion .mat files found in the directory.
        ValueError: Number of motion matrices does not match STC volumes.
    """
    bold_ref_img = nib.nifti1.load(bold_ref)
    template_img = nib.nifti1.load(template)

    # If 4D, extract first volume
    if len(bold_ref_img.shape) > 3:
        bold_ref_img = bold_ref_img[..., 0]
    # If same shape, no need to resample
    if bold_ref_img.shape == template_img.shape:
        return template_img

    template_img = resample_from_to(template_img, bold_ref_img)
    template_img_path = (
        generate_exec_folder("template_resample_to_bold_grid")
        / "template_resampled.nii.gz"
    )
    nib.save(template_img, template_img_path)
    return template_img_path
