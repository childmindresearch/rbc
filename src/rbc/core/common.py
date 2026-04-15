"""Processing steps shared across anatomical and functional streams.

Currently provides:
- Deobliquing and RPI reorientation (initial preprocessing for T1w and BOLD).
- 4D NIfTI splitting and merging utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import styxcache
from niwrap import afni

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

from rbc.core.fileops import file_tmp_copy
from rbc.core.nifti import strip_afni_volatile_metadata
from rbc.core.niwrap import generate_exec_folder

__all__ = ["deoblique_and_reorient", "merge_3d_to_4d", "split_4d"]


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
        # 3drefit mutates in place, and styxcache does not replay
        # mutable-input mutations on cache hits. Bypass it so it always runs,
        # then strip AFNI's non-deterministic extension (timestamps + random
        # UUID) so the downstream cached 3dresample call keys on stable bytes.
        with styxcache.bypass():
            afni.v_3drefit(in_file=tmp_file, deoblique=True)
        strip_afni_volatile_metadata(tmp_file)
        return afni.v_3dresample(
            in_file=tmp_file, prefix=output_fname, orientation="RPI"
        )


def split_4d(img_4d: Path) -> list[Path]:
    """Split a 4D NIfTI timeseries into individual 3D volumes.

    Volumes are written as uncompressed NIfTI (.nii) to avoid gzip
    overhead on float intermediates that are read back immediately.

    Args:
        img_4d: Path to a 4D NIfTI image.

    Returns:
        Sorted list of paths to the individual 3D volume files.
    """
    img = nib.nifti1.load(img_4d)
    volumes = nib.four_to_three(img)
    out_dir = generate_exec_folder(suffix="split4d")
    paths: list[Path] = []
    for idx, vol in enumerate(volumes):
        out_path = out_dir / f"vol_{idx:04d}.nii"
        nib.save(vol, out_path)
        paths.append(out_path)
    return paths


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
