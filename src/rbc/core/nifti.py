"""Low-level NIfTI helpers.

Lightweight functions for querying NIfTI metadata (e.g. number of volumes)
without loading the full image data into memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["nifti_num_volumes"]


def nifti_num_volumes(in_file: str | Path) -> int:
    """Return the number of volumes in a NIfTI image (returns 1 for 3-D images).

    Used to find the middle volume index for motion-reference extraction.
    """

    shape = nibabel.nifti1.load(in_file).shape
    return shape[3] if len(shape) > 3 else 1
