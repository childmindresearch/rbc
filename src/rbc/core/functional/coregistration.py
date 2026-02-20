"""BOLD to T1w coregistration using boundary-based registration.

The BOLD reference image is registered to the anatomical T1w space using a
two-stage approach. First, an initial linear registration with correlation
ratio cost function establishes a rough alignment. This is then refined using
boundary-based registration (BBR). The resulting transformation matrix is used
downstream for resampling the BOLD timeseries to template space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import fsl

if TYPE_CHECKING:
    from pathlib import Path

from rbc.core.resources import FSL_RESOURCES


def coregister_bold_to_t1w(
    in_file: Path,
    reference: Path,
    wm_seg: Path,
) -> fsl.FlirtOutputs:
    """Align BOLD reference to T1w anatomical space using BBR.

    An initial linear registration (correlation ratio) is refined using
    boundary-based registration (BBR), using the white matter boundary
    contrast for precise alignment.

    Args:
        in_file: Skull-stripped BOLD reference image.
        reference: T1w brain-extracted anatomical reference.
        wm_seg: White matter segmentation mask for BBR.

    Returns:
        Affine transformation matrix from BOLD to T1w space
        (use ``.out_matrix_file`` for the matrix)
    """
    # Step 1: Initial linear registration with correlation ratio
    linear_result = fsl.flirt(
        in_file=in_file,
        reference=reference,
        out_file="linear_bold_to_t1w.nii.gz",
        out_matrix_file="linear_bold_to_t1w.mat",
        cost="corratio",
        dof=6,
        interp="trilinear",
    )

    # Step 2: BBR refinement using WM boundary
    return fsl.flirt(
        in_file=in_file,
        reference=reference,
        out_file="bbr_bold_to_t1w.nii.gz",
        out_matrix_file="bbr_bold_to_t1w.mat",
        cost="bbr",
        wm_seg=wm_seg,
        dof=6,
        in_matrix_file=linear_result.out_matrix_file,
        schedule=FSL_RESOURCES.bbr_schedule,
    )
