"""Full pipeline test for longitudinal QC.

Verifies that ``rbc longitudinal qc`` produces a registration-quality
TSV with Dice/Jaccard metrics and a pass/fail flag.
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


_SUB = "01"
_SES = "test"
_TASK = "fingerfootlips"
_STEM = f"sub-{_SUB}_ses-{_SES}_task-{_TASK}"


def _file_tree(root: Path) -> str:
    files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    return "\n".join(str(f) for f in files) if files else "(empty)"


@pytest.mark.slow
def test_longitudinal_qc_tsv_exists(
    longitudinal_pipeline_data: Path,
) -> None:
    """QC stage writes a registration quality TSV."""
    func = longitudinal_pipeline_data / f"sub-{_SUB}" / f"ses-{_SES}" / "func"
    tree = _file_tree(longitudinal_pipeline_data)

    qc_files = list(func.glob(f"{_STEM}_space-longitudinal_*quality*.tsv"))
    assert qc_files, f"No QC quality TSV found\n--- file tree ---\n{tree}"


@pytest.mark.slow
def test_longitudinal_qc_tsv_has_expected_columns(
    longitudinal_pipeline_data: Path,
) -> None:
    """QC TSV contains dice, jaccard, coverage, cross_corr, passed columns."""
    func = longitudinal_pipeline_data / f"sub-{_SUB}" / f"ses-{_SES}" / "func"
    qc_files = list(func.glob(f"{_STEM}_space-longitudinal_*quality*.tsv"))
    assert qc_files, "No QC quality TSV found"

    with qc_files[0].open() as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)

    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"
    expected_cols = {
        "sub",
        "ses",
        "task",
        "run",
        "dice",
        "jaccard",
        "coverage",
        "cross_corr",
        "passed",
    }
    assert expected_cols.issubset(rows[0].keys()), (
        f"Missing columns: {expected_cols - rows[0].keys()}"
    )

    # Dice should be a reasonable value (> 0 at minimum)
    dice = float(rows[0]["dice"])
    assert dice > 0.0, f"Dice coefficient is {dice}, expected > 0"
