"""Full pipeline test for ``rbc longitudinal all``.

Verifies that the combined pipeline produces the same outputs as running
each stage individually, mirroring tests/integration/test_all.py for the
cross-sectional pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


_SUB = "01"
_SES = "test"
_TASK = "fingerfootlips"
_STEM = f"sub-{_SUB}_ses-{_SES}_task-{_TASK}"


def _relative_files(root: Path) -> set[str]:
    """Return the set of file paths relative to *root*."""
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def _file_tree(root: Path) -> str:
    files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    return "\n".join(str(f) for f in files) if files else "(empty)"


def _longitudinal_files(root: Path) -> set[str]:
    """Return only longitudinal-space files relative to *root*."""
    return {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and "longitudinal" in p.name
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_longitudinal_all_produces_derivatives(
    longitudinal_all_data: Path,
) -> None:
    """``rbc longitudinal all`` runs end-to-end and writes expected files."""
    func = longitudinal_all_data / f"sub-{_SUB}" / f"ses-{_SES}" / "func"
    tree = _file_tree(longitudinal_all_data)

    # Functional derivatives in longitudinal space
    assert list(func.glob(f"{_STEM}_space-longitudinal_desc-preproc_bold.nii.gz")), (
        f"Missing longitudinal BOLD\n--- file tree ---\n{tree}"
    )

    # Metrics
    assert list(func.glob(f"{_STEM}_space-longitudinal_*_alff.nii.gz")), (
        f"Missing ALFF\n--- file tree ---\n{tree}"
    )
    assert list(func.glob(f"{_STEM}_space-longitudinal_*_timeseries.tsv")), (
        f"Missing timeseries\n--- file tree ---\n{tree}"
    )

    # QC
    assert list(func.glob(f"{_STEM}_space-longitudinal_*quality*.tsv")), (
        f"Missing QC TSV\n--- file tree ---\n{tree}"
    )


@pytest.mark.slow
def test_longitudinal_all_vs_sequential_filenames_match(
    longitudinal_pipeline_data: Path,
    longitudinal_all_data: Path,
) -> None:
    """Both invocation styles must produce the same longitudinal-space files.

    Compares only ``*longitudinal*`` filenames (not cross-sectional
    derivatives, which may differ in tmp path layout).
    """
    seq_files = _longitudinal_files(longitudinal_pipeline_data)
    all_files = _longitudinal_files(longitudinal_all_data)

    missing_from_all = seq_files - all_files
    extra_in_all = all_files - seq_files

    assert not missing_from_all, (
        "Files produced by sequential run but missing from 'rbc longitudinal all':\n"
        + "\n".join(sorted(missing_from_all))
    )
    assert not extra_in_all, (
        "Files produced by 'rbc longitudinal all' but missing from sequential run:\n"
        + "\n".join(sorted(extra_in_all))
    )
