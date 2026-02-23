"""Processing steps shared across anatomical and functional streams.

Currently provides:
- Deobliquing and RPI reorientation (initial preprocessing for T1w and BOLD).
- Transformation conversion between FSL (.mat) and ITK (.txt) formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni
from niwrap_c3d import c3d

if TYPE_CHECKING:
    from pathlib import Path

from rbc.core.fileops import file_tmp_copy

__all__ = ["deoblique_and_reorient", "mat_to_itk"]


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


def mat_to_itk(mat: Path, reference: Path, source: Path, output: str) -> Path:
    """Convert a .mat affine to ITK compatible .txt format.

    Args:
        mat: Path to the input FSL-style affine matrix (.mat).
        reference: Path to the reference (fixed) image volume.
        source: Path to the source (moving) image volume.
        output: Filename or path for the resulting ITK transformation file (.txt).

    Returns:
        ITK-compatible transformation file.
    """
    result = c3d.c3d_affine_tool(
        reference_file=reference,
        source_file=source,
        transform_file=mat,
        out_itk_transform=output,
        fsl2ras=True,
    )
    return result.itk_transform_outfile
