"""Mask erosion utilities for nuisance regression.

Provides iterative binary erosion of tissue masks (CSF, WM, brain) to
reduce partial-volume contamination before extracting nuisance signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

if TYPE_CHECKING:
    from pathlib import Path


def erode_mask_to_proportion(mask: np.ndarray, target_proportion: float) -> np.ndarray:
    """Iteratively erode a binary mask until its volume reaches a target fraction.

    Uses 3-D binary erosion with a 3x3x3 cross structuring element,
    repeating until the remaining volume is at or below
    ``target_proportion * original_volume``.

    Args:
        mask: 3-D boolean-compatible array.
        target_proportion: Fraction of the original mask volume to retain
            (e.g. 0.9 for CSF, 0.6 for WM).  Must be in ``(0, 1)``.

    Returns:
        3-D boolean array of the eroded mask.

    Raises:
        ValueError: If *mask* is not 3-D or *target_proportion* is out of range.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D")
    if not 0 < target_proportion < 1:
        raise ValueError(
            f"target_proportion must be in (0, 1), got {target_proportion}"
        )

    mask_bool = mask > 0
    original_volume = mask_bool.sum()

    if original_volume == 0:
        return mask_bool

    target_volume = original_volume * target_proportion
    eroded = mask_bool.copy()

    while eroded.sum() > target_volume:
        candidate = binary_erosion(eroded)
        if not np.any(candidate):
            break
        # Pick whichever is closer to the target: before or after this step
        overshoot = target_volume - candidate.sum()
        undershoot = eroded.sum() - target_volume
        if overshoot > 0 and overshoot > undershoot:
            break
        eroded = candidate

    return eroded


def erode_mask_by_distance(
    mask: np.ndarray,
    voxel_sizes: tuple[float, float, float],
    distance_mm: float,
) -> np.ndarray:
    """Erode a mask by a fixed Euclidean distance in millimetres.

    Uses a Euclidean distance transform that accounts for anisotropic
    voxel sizes, then thresholds to keep only voxels whose distance to
    the mask boundary is at least *distance_mm*.

    Args:
        mask: 3-D boolean-compatible array.
        voxel_sizes: Voxel dimensions ``(dx, dy, dz)`` in mm.
        distance_mm: Desired erosion distance in mm.  Must be positive.

    Returns:
        3-D boolean array of the eroded mask.

    Raises:
        ValueError: If *mask* is not 3-D or *distance_mm* is not positive.
    """
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D")
    if distance_mm <= 0:
        raise ValueError(f"distance_mm must be positive, got {distance_mm}")

    mask_bool = mask > 0
    dist = distance_transform_edt(mask_bool, sampling=voxel_sizes)
    return dist >= distance_mm


def erode_csf_mask(mask: np.ndarray) -> np.ndarray:
    """Erode a CSF mask to 90 % of its original volume."""
    return erode_mask_to_proportion(mask, target_proportion=0.9)


def erode_wm_mask(mask: np.ndarray) -> np.ndarray:
    """Erode a WM mask to 60 % of its original volume."""
    return erode_mask_to_proportion(mask, target_proportion=0.6)


def erode_brain_mask(
    mask: np.ndarray,
    voxel_sizes: tuple[float, float, float],
) -> np.ndarray:
    """Erode a brain mask by 30 mm."""
    return erode_mask_by_distance(mask, voxel_sizes, distance_mm=30.0)


def create_union_mask(
    mask_a_file: str | Path,
    mask_b_file: str | Path,
) -> Path:
    """Create the logical OR (union) of two binary masks.

    Args:
        mask_a_file: Path to first 3-D binary mask.
        mask_b_file: Path to second 3-D binary mask.

    Returns:
        Path to the union mask NIfTI file.
    """
    import nibabel as nib

    from rbc.core.niwrap import generate_exec_folder

    out_dir = generate_exec_folder("union_mask")

    img_a = nib.nifti1.load(mask_a_file)
    data_a = img_a.get_fdata() > 0
    data_b = nib.nifti1.load(mask_b_file).get_fdata() > 0

    union = (data_a | data_b).astype(np.uint8)
    out_path = out_dir / "union_mask.nii.gz"
    nib.nifti1.Nifti1Image(union, img_a.affine, img_a.header).to_filename(str(out_path))
    return out_path


class ErodedMasks(NamedTuple):
    """Paths to eroded tissue masks."""

    csf: Path
    wm: Path
    brain: Path


def compute_eroded_masks(
    csf_file: str | Path,
    wm_file: str | Path,
    brain_file: str | Path,
) -> ErodedMasks:
    """Load tissue masks, erode them, and write results to disk.

    Args:
        csf_file: Path to 3-D CSF probability/binary mask.
        wm_file: Path to 3-D WM probability/binary mask.
        brain_file: Path to 3-D brain mask.

    Returns:
        :class:`ErodedMasks` with paths to the eroded NIfTI files.
    """
    import nibabel as nib

    from rbc.core.niwrap import generate_exec_folder

    out_dir = generate_exec_folder("erode_masks")

    csf_img = nib.nifti1.load(csf_file)
    csf_eroded = erode_csf_mask(csf_img.get_fdata())
    csf_out = out_dir / "csf_eroded.nii.gz"
    nib.nifti1.Nifti1Image(
        csf_eroded.astype(np.uint8), csf_img.affine, csf_img.header
    ).to_filename(str(csf_out))

    wm_img = nib.nifti1.load(wm_file)
    wm_eroded = erode_wm_mask(wm_img.get_fdata())
    wm_out = out_dir / "wm_eroded.nii.gz"
    nib.nifti1.Nifti1Image(
        wm_eroded.astype(np.uint8), wm_img.affine, wm_img.header
    ).to_filename(str(wm_out))

    brain_img = nib.nifti1.load(brain_file)
    voxel_sizes = tuple(float(v) for v in brain_img.header.get_zooms()[:3])
    brain_eroded = erode_brain_mask(brain_img.get_fdata(), voxel_sizes)  # type: ignore[arg-type]
    brain_out = out_dir / "brain_eroded.nii.gz"
    nib.nifti1.Nifti1Image(
        brain_eroded.astype(np.uint8), brain_img.affine, brain_img.header
    ).to_filename(str(brain_out))

    return ErodedMasks(csf=csf_out, wm=wm_out, brain=brain_out)
