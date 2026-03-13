"""Processing steps shared across anatomical and functional streams.

Currently provides:
- Deobliquing and RPI reorientation (initial preprocessing for T1w and BOLD).
- Transformation conversion between FSL (.mat) and ITK (.txt) formats.
- 4D NIfTI splitting and merging utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
from niwrap import afni, c3d, fsl

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from rbc.core.fileops import file_tmp_copy

__all__ = ["deoblique_and_reorient", "mat_to_itk", "merge_3d_to_4d", "split_4d"]


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
    assert result.itk_transform_outfile is not None  # noqa: S101
    return result.itk_transform_outfile


def split_4d(img_4d: Path) -> list[Path]:
    """Split a 4D NIfTI timeseries into individual 3D volumes.

    Args:
        img_4d: Path to a 4D NIfTI image.

    Returns:
        Sorted list of paths to the individual 3D volume files.
    """
    split_result = fsl.fslsplit(
        infile=img_4d, separation_time=True, output_basename="vol_"
    )
    assert split_result.out_files is not None  # noqa: S101
    out_files = split_result.out_files
    out_dir = out_files[0].parent if isinstance(out_files, list) else out_files.parent
    return sorted(out_dir.glob("vol_*.nii.gz"))


def merge_3d_to_4d(volumes: Sequence[Path], output: Path) -> Path:
    """Merge a sequence of 3D NIfTI volumes into a single 4D timeseries.

    Args:
        volumes: Ordered sequence of paths to 3D NIfTI images.
        output: Path to write the merged 4D image.

    Returns:
        Path to the merged 4D NIfTI image.
    """
    imgs = [nib.nifti1.load(v) for v in volumes]
    merged = nib.funcs.concat_images(imgs, axis=None)
    nib.save(merged, output)
    return output
