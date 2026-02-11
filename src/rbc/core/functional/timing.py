"""Slice Timing Correction.

Since slices are acquired sequentially rather than simultaneously,
each slice represents a slightly different timepoint. AFNI ``3dTshift``
temporally interpolates all slices to a common reference time to
produce a timeseries where all slices within each volume represent
the same temporal moment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

if TYPE_CHECKING:
    from pathlib import Path


def slice_timing_correction(
    in_file: Path, tr: float, t_pattern: str | None = None, tzero: str | None = None
) -> afni.V3dTshiftOutputs:
    """Apply slice timing correction to a BOLD timeseries.

    Temporally interpolates slices to align them to a common reference time,
    correcting for the sequential acquisition of slices within each volume.
    If tpattern is not provided, AFNI will attempt to detect from the image
    header.

    Args:
        in_file: Truncated BOLD timeseries to correct.
        tr: Repetition time in seconds (e.g., 2.0).
        t_pattern: Slice acquisition pattern (e.g., 'alt+z', 'seq+z').
            If None, auto-detected from header.
        tzero: Time in seconds to align slices to. If None, uses first slice.

    Returns:
        AFNI 3dTshift outputs (use ``.out_file`` for corrected timeseries).
    """
    return afni.v_3d_tshift(
        in_file=in_file,
        tr=tr,
        tpattern=t_pattern if t_pattern is not None else None,
        tzero=tzero if tzero is not None else None,
        prefix="stc.nii.gz",
    )
