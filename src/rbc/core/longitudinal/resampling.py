"""Resampling utilities for longitudinal templates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
from nibabel.processing import resample_from_to

if TYPE_CHECKING:
    from pathlib import Path

from rbc.core.niwrap import generate_exec_folder


def resample_img_to_bold_grid(bold_ref: Path, img: Path) -> Path:
    """Resample template to BOLD grid if shapes differ.

    Args:
        bold_ref: BOLD reference volume.
        img: 3D image in target space to resample.

    Returns:
        Resampled 3D image with BOLD grid
    """
    bold_ref_img = nib.nifti1.load(bold_ref)
    img_obj = nib.nifti1.load(img)

    # If 4D, extract first volume
    if len(bold_ref_img.shape) > 3:
        bold_ref_img = nib.four_to_three(bold_ref_img.slicer[..., 0])[0]
    # If same shape, no need to resample
    if bold_ref_img.shape == img_obj.shape:
        return img

    img_resampled = resample_from_to(img_obj, bold_ref_img)
    img_resampled_path = (
        generate_exec_folder("img_resample_to_bold_grid") / "resampled.nii.gz"
    )
    nib.save(img_resampled, img_resampled_path)
    return img_resampled_path
