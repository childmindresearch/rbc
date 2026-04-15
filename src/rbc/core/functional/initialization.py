"""BOLD initialization steps.

After reorientation (handled in ``rbc.core.common``), BOLD data undergoes an
initialization step before motion correction: TR truncation -- discard the
first *N* volumes (default 2) to allow the scanner signal to reach steady
state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from niwrap import afni

if TYPE_CHECKING:
    from pathlib import Path


def truncate_trs(in_file: Path, start_tr: int) -> Path:
    """Discard the first *N* TRs from a BOLD timeseries.

    Early volumes are typically discarded because the MR signal has not yet
    reached a steady state, which would introduce intensity artifacts.

    Args:
        in_file: Reoriented BOLD timeseries.
        start_tr: Number of initial TRs to drop (e.g. 2).

    Returns:
        Path to the truncated BOLD timeseries.
    """
    result = afni.v_3dcalc(
        dataset_a=afni.v_3dcalc_dataset_a_file(
            file=in_file, selectors_=f"[{start_tr}..$]"
        ),
        expression="a",
        prefix="truncated.nii.gz",
    )
    assert result.output_file is not None  # noqa: S101
    return result.output_file
