"""Binary mask erosion primitives.

Provides iterative and distance-based erosion of 3-D binary masks.
Convenience wrappers apply C-PAC default parameters for CSF, WM, and
brain masks used during nuisance regression.

Note:
    C-PAC's RBC pipeline config (``pipeline_config_rbc-options.yml``)
    sets ``erode_mask: Off`` for all tissue regressors. The erosion
    functions here are retained as general-purpose utilities and for
    optional use when erosion is explicitly enabled.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


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
    """Erode a CSF mask to 90 % of its original volume.

    Matches C-PAC default ``csf_erosion_prop: 0.9``.  Note that
    Ciric et al. (2017) used 1-voxel erosion and Behzadi et al. (2007)
    applied no CSF erosion.
    """
    return erode_mask_to_proportion(mask, target_proportion=0.9)


def erode_wm_mask(mask: np.ndarray) -> np.ndarray:
    """Erode a WM mask to 60 % of its original volume.

    Matches C-PAC default ``wm_erosion_prop: 0.6``.  Note that
    Ciric et al. (2017) used 2-voxel erosion and Behzadi et al. (2007)
    used 2-voxel erosion with a partial-volume threshold of 0.99.
    """
    return erode_mask_to_proportion(mask, target_proportion=0.6)


def erode_brain_mask(
    mask: np.ndarray,
    voxel_sizes: tuple[float, float, float],
) -> np.ndarray:
    """Erode a brain mask by 30 mm.

    Matches C-PAC default ``brain_mask_erosion_mm: 30``.  This erosion
    is only relevant for tCompCor (Behzadi et al. 2007), which is not
    currently implemented in RBC.  It should *not* be used for global
    signal extraction (Ciric et al. 2017 use the uneroded brain mask).
    """
    return erode_mask_by_distance(mask, voxel_sizes, distance_mm=30.0)
