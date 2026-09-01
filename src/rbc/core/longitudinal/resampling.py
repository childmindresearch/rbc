"""Resampling utilities for longitudinal templates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

if TYPE_CHECKING:
    from pathlib import Path

from rbc.core.niwrap import generate_exec_folder


def resample_img_to_bold_res(bold_ref: Path, img: Path, order: int = 3) -> Path:
    """Resample *img* to the BOLD reference's resolution, keeping *img*'s origin.

    Only the voxel spacing (resolution) is taken from the BOLD reference. The
    output keeps *img*'s affine orientation and origin, so the brain content
    is re-tiled at the new resolution in *img*'s own coordinate space rather
    than being shifted onto the BOLD's (generally different) volume corner.

    Args:
        bold_ref: BOLD reference volume; its voxel spacing defines the target
            resolution.
        img: 3D image in target space to resample.
        order: Interpolation order used during resampling.

    Returns:
        Resampled 3D image on *img*'s grid at the BOLD's resolution. If *img*
        is already at the BOLD's resolution, *img* is returned unchanged.
    """
    bold_ref_img = nib.nifti1.load(bold_ref)
    img_obj = nib.nifti1.load(img)

    # Voxel sizes from the affine (column norms, as in Volume.voxel_sizes)
    # rather than header pixdim, since the resample below is affine-driven.
    bold_zooms = np.linalg.norm(bold_ref_img.affine[:3, :3], axis=0)
    in_zooms = np.linalg.norm(img_obj.affine[:3, :3], axis=0)
    if np.allclose(in_zooms, bold_zooms, rtol=1e-5, atol=1e-6):
        return img

    # Rescale each voxel to the BOLD spacing, keeping *img*'s direction/origin.
    target_affine = np.array(img_obj.affine, dtype=float)
    target_affine[:3, :3] *= bold_zooms / in_zooms

    # Re-tile the same physical extent at the new spacing (axis-aligned grids).
    in_shape = np.asarray(img_obj.shape[:3], dtype=float)
    target_shape = (np.ceil((in_shape - 1.0) * in_zooms / bold_zooms) + 1).astype(int)

    img_resampled = resample_from_to(
        img_obj, (tuple(target_shape), target_affine), order=order
    )
    img_resampled_path = (
        generate_exec_folder("img_resample_to_bold_res") / "resampled.nii.gz"
    )
    nib.save(img_resampled, img_resampled_path)
    return img_resampled_path
