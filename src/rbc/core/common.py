"""General functions useful across modalities."""

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
    """AFNI deobliquing and reorientation to RPI.

    Sets image into a cardinal orientation if it was acquired obliquely from scanner
    and standardize orientation of images ('RPI' is internal assumption from AFNI).

    Args:
        in_file: Input T1w to reorient
        output_fname: Output filename

    Returns:
        An object representing the outputs from AFNI's 3D resample.
    """
    with file_tmp_copy(in_file) as tmp_file:
        afni.v_3drefit(in_file=tmp_file, deoblique=True)
        return afni.v_3dresample(
            in_file=tmp_file, prefix=output_fname, orientation="RPI"
        )
