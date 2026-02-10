"""Preprocessing steps shared across anatomical and functional streams.

Currently provides deobliquing and RPI reorientation, which is the first
step applied to both T1w and BOLD images.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

if TYPE_CHECKING:
    from pathlib import Path

from rbc.core.fileops import file_tmp_copy

__all__ = ["deoblique_and_reorient"]


def deoblique_and_reorient(
    in_file: Path, output_fname: str = "reoriented.nii.gz"
) -> afni.V3dresampleOutputs:
    """Deoblique and reorient an image to RPI orientation.

    Many scanners acquire images at an oblique angle, producing a non-cardinal
    orientation matrix. This step removes the oblique transform (``3drefit
    -deoblique``) and resamples to a standard Right-Posterior-Inferior (RPI)
    orientation (``3dresample``), which AFNI tools assume internally. Applied as
    the first step to both anatomical and functional inputs.

    Args:
        in_file: Image to reorient (T1w or BOLD).
        output_fname: Output filename.

    Returns:
        AFNI 3dresample outputs (use ``.out_file`` for the reoriented image).
    """
    with file_tmp_copy(in_file) as tmp_file:
        afni.v_3drefit(in_file=tmp_file, deoblique=True)
        return afni.v_3dresample(
            in_file=tmp_file, prefix=output_fname, orientation="RPI"
        )
