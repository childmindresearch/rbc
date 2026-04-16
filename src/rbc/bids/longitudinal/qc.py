"""BIDS resolve and export for the longitudinal QC workflow.

Longitudinal QC is minimal: Dice/Jaccard overlap between the anatomical
brain mask and BOLD brain mask, both in longitudinal template space.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from rbc.bids import Suffix

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl

    from rbc.bids import Bids


class QCLongInputs(TypedDict):
    """Resolved inputs for the longitudinal registration QC."""

    anat_brain_mask: Path
    bold_mask: Path


class LongitudinalQCOutputs(NamedTuple):
    """QC outputs from the longitudinal registration overlap check.

    Attributes:
        dice: Dice coefficient between anat and BOLD masks.
        jaccard: Jaccard index between anat and BOLD masks.
        coverage: Coverage of the smaller mask by the overlap.
        cross_corr: Pearson correlation between flattened masks.
        passed: Whether the run passes the Dice threshold.
        qc_file: Path to the written single-row QC TSV.
    """

    dice: float
    jaccard: float
    coverage: float
    cross_corr: float
    passed: bool
    qc_file: Path


def resolve_longitudinal_qc(
    anat_long_q: Bids,
    func_long_q: Bids,
    anat_long_df: pl.DataFrame,
    func_long_df: pl.DataFrame,
) -> QCLongInputs:
    """Resolve longitudinal-space masks for registration QC.

    Args:
        anat_long_q: Bids builder for ``space=longitudinal`` anat queries.
        func_long_q: Bids builder for ``space=longitudinal`` func queries.
        anat_long_df: DataFrame of longitudinal anatomical derivatives.
        func_long_df: DataFrame of longitudinal functional derivatives.

    Returns:
        Dict with ``anat_brain_mask`` and ``bold_mask`` paths.
    """
    return {
        "anat_brain_mask": anat_long_q.expect(
            anat_long_df, suffix=Suffix.MASK, desc="T1w"
        ),
        "bold_mask": func_long_q.expect(func_long_df, suffix=Suffix.MASK, desc="brain"),
    }


def export_longitudinal_qc(
    func_long: Bids,
    outputs: LongitudinalQCOutputs,
) -> None:
    """Export longitudinal QC results as a single-row TSV.

    Args:
        func_long: Bids builder with ``space="longitudinal"`` for func.
        outputs: QC overlap metrics.
    """
    func_long.save(
        outputs.qc_file,
        suffix="quality",
        desc="registration",
        extension=".tsv",
    )


def write_longitudinal_qc_tsv(
    out_path: Path,
    *,
    sub: str,
    ses: str,
    task: str,
    run: int | str,
    dice: float,
    jaccard: float,
    coverage: float,
    cross_corr: float,
    passed: bool,
) -> Path:
    """Write a single-row longitudinal QC TSV.

    Args:
        out_path: Destination file path.
        sub: Subject ID.
        ses: Session label.
        task: Task label.
        run: Run number.
        dice: Dice coefficient.
        jaccard: Jaccard index.
        coverage: Coverage metric.
        cross_corr: Cross-correlation.
        passed: Pass/fail flag.

    Returns:
        The written file path.
    """
    fieldnames = [
        "sub",
        "ses",
        "task",
        "run",
        "dice",
        "jaccard",
        "coverage",
        "cross_corr",
        "passed",
    ]
    row = {
        "sub": sub,
        "ses": ses,
        "task": task,
        "run": run,
        "dice": f"{dice:.6f}",
        "jaccard": f"{jaccard:.6f}",
        "coverage": f"{coverage:.6f}",
        "cross_corr": f"{cross_corr:.6f}",
        "passed": str(passed),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    return out_path
