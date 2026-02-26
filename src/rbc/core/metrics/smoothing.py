"""Spatial smoothing of derivative maps.

Applies iterative Gaussian smoothing via AFNI ``3dBlurToFWHM`` to bring
derivative maps (ALFF, fALFF, ReHo, centrality) to a target spatial
smoothness within a brain mask.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

if TYPE_CHECKING:
    from pathlib import Path


def smooth(
    in_file: Path,
    mask_file: Path,
    fwhm: float = 6.0,
) -> Path:
    """Spatially smooth a 3D map to a target FWHM within a brain mask.

    Uses AFNI 3dBlurToFWHM to iteratively blur the input until the
    estimated smoothness reaches the requested FWHM.

    Args:
        in_file: 3D NIfTI derivative map to smooth.
        mask_file: Binary brain mask; voxels outside are set to zero.
        fwhm: Target full-width at half-maximum in mm.

    Returns:
        Path to the smoothed map.
    """
    result = afni.v_3d_blur_to_fwhm(
        in_file=in_file,
        mask=mask_file,
        fwhm=fwhm,
        prefix="smoothed.nii.gz",
    )
    assert result.out_file is not None  # noqa: S101
    return result.out_file
