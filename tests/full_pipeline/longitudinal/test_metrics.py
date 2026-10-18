"""Full pipeline test for longitudinal metrics.

Verifies that ``rbc longitudinal metrics`` produces ALFF, fALFF, ReHo,
and atlas timeseries/correlation outputs in ``space-longitudinal``.
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


def _file_tree(root: Path) -> str:
    files = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    return "\n".join(str(f) for f in files) if files else "(empty)"


@pytest.mark.slow
def test_longitudinal_metrics_produces_expected_files(
    longitudinal_pipeline_data: Path,
) -> None:
    """Metrics stage writes ALFF, fALFF, ReHo, timeseries, and correlations."""
    func = longitudinal_pipeline_data / f"sub-{_SUB}" / f"ses-{_SES}" / "func"
    tree = _file_tree(longitudinal_pipeline_data)

    expected_fragments = [
        # raw maps — always produced
        f"{_STEM}_space-longitudinal_reg-36parameter_alff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_falff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_reho.nii.gz",
        # z-scored raw maps - always produced
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-zstd_alff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-zstd_falff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-zstd_reho.nii.gz",
        # smoothed + z-scored smoothed — produced with --smooth 6
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-sm6_alff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-sm6_falff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-sm6_reho.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-sm6Zstd_alff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-sm6Zstd_falff.nii.gz",
        f"{_STEM}_space-longitudinal_reg-36parameter_desc-sm6Zstd_reho.nii.gz",
    ]
    for name in expected_fragments:
        assert (func / name).is_file(), (
            f"Missing metrics file: {name}\n--- file tree ---\n{tree}"
        )


@pytest.mark.slow
def test_longitudinal_metrics_timeseries_exist(
    longitudinal_pipeline_data: Path,
) -> None:
    """Atlas timeseries and correlation TSVs are produced."""
    func = longitudinal_pipeline_data / f"sub-{_SUB}" / f"ses-{_SES}" / "func"
    tree = _file_tree(longitudinal_pipeline_data)

    timeseries = list(func.glob(f"{_STEM}_space-longitudinal_*_timeseries.tsv"))
    assert timeseries, f"No timeseries TSV found\n--- file tree ---\n{tree}"

    correlations = list(func.glob(f"{_STEM}_space-longitudinal_*_correlations.tsv"))
    assert correlations, f"No correlation TSV found\n--- file tree ---\n{tree}"
