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

from rbc.core.niwrap import generate_exec_folder

if TYPE_CHECKING:
    from pathlib import Path


def slice_timing_correction(
    in_file: Path,
    tr: float | None = None,
    tpattern: str | list[float] | None = None,
) -> afni.V3dTshiftOutputs:
    """Apply slice timing correction to a BOLD timeseries.

    Temporally interpolates slices to align them to a common reference time,
    correcting for the sequential acquisition of slices within each volume.

    Args:
        in_file: Truncated BOLD timeseries to correct.
        tr: Repetition time in seconds (e.g., 2.0).
        tpattern: Slice acquisition pattern.

    Returns:
        AFNI 3dTshift outputs (use ``.out_file`` for corrected timeseries).
    """
    if isinstance(tpattern, str):
        tpattern_arg = afni.v_3d_tshift_tpattern_mode_string(tpattern_string=tpattern)

    elif isinstance(tpattern, list):
        exec_dir = generate_exec_folder()
        timing_file = exec_dir / "SliceTiming.1D"
        timing_file.write_text("\n".join(map(str, tpattern)))
        tpattern_arg = afni.v_3d_tshift_tpattern_mode_file(
            tpattern_file=str(timing_file)
        )

    else:
        tpattern_arg = None

    return afni.v_3d_tshift(
        in_file=in_file,
        prefix="stc.nii.gz",
        tr=afni.v_3d_tshift_tr_microsyntax(value=tr, unit="s") if tr else None,
        tpattern=tpattern_arg if tpattern else None,
    )
