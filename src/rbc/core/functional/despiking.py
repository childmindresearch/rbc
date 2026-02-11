"""Despiking of BOLD timeseries.

Despiking identifies and attenuates transient signal spikes in BOLD data
that can arise from scanner artifacts or subject motion. Using AFNI
``3dDespike``, outliers in each voxel's timeseries are detected and replaced
with interpolated values, producing a cleaned timeseries for further preprocessing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

if TYPE_CHECKING:
    from pathlib import Path


def despike_bold(in_file: Path) -> afni.V3dDespikeOutputs:
    """Remove temporal outliers (spikes) from a BOLD timeseries.

    Identifies voxel timeseries values that deviate substantially from
    their local temporal neighborhood and replaces them with interpolated values
    that better fit the local temporal structure.

    Args:
        in_file: BOLD timeseries.

    Returns:
        AFNI 3dDespike outputs (use ``.out_file`` for the despiked series).
    """
    return afni.v_3d_despike(
        in_file=in_file,
        prefix="despiked.nii.gz",
    )
