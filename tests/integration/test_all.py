"""Integration test for ``rbc all`` pipeline stage handoff.

Runs the full pipeline on the ds000001 test dataset with Docker and verifies
that it completes without errors and that key derivative files exist.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_TEST_DATASET = Path(__file__).parents[1] / "data" / "ds000001"

# Subject with no session in ds000001.
_SUB = "01"
_TASK = "balloonanalogrisktask"
_RUN = "01"


@pytest.mark.slow
def test_rbc_all_completes_and_produces_derivatives(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """``rbc all`` runs end-to-end and writes expected derivative files."""
    runner = request.config.getoption("--runner")
    output_dir = tmp_path / "derivatives"
    output_dir.mkdir()

    rbc_exe = shutil.which("rbc")
    assert rbc_exe is not None, "rbc CLI not found on PATH"

    result = subprocess.run(  # noqa: S603
        [
            rbc_exe,
            "all",
            str(_TEST_DATASET),
            "-o",
            str(output_dir),
            "--participant-label",
            _SUB,
            "--runner",
            runner,
        ],
        capture_output=True,
        text=True,
        timeout=7200,  # 2 h ceiling
    )
    assert result.returncode == 0, (
        f"rbc all exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )

    sub_dir = output_dir / f"sub-{_SUB}"

    # -- Dataset-level metadata --
    assert (output_dir / "dataset_description.json").is_file()

    # -- Anatomical derivatives --
    anat = sub_dir / "anat"
    anat_files = [
        f"sub-{_SUB}_desc-brain_T1w.nii.gz",
        f"sub-{_SUB}_desc-T1w_mask.nii.gz",
        f"sub-{_SUB}_desc-csf_mask.nii.gz",
        f"sub-{_SUB}_desc-gm_mask.nii.gz",
        f"sub-{_SUB}_desc-wm_mask.nii.gz",
        f"sub-{_SUB}_desc-wmBBR_mask.nii.gz",
    ]
    for name in anat_files:
        assert (anat / name).is_file(), f"Missing anatomical file: {name}"

    # -- Functional derivatives --
    func = sub_dir / "func"
    bold_stem = f"sub-{_SUB}_task-{_TASK}_run-{_RUN}"
    func_files = [
        f"{bold_stem}_sbref.nii.gz",
        f"{bold_stem}_desc-preproc_bold.nii.gz",
        f"{bold_stem}_desc-motionParams_motion.1D",
        f"{bold_stem}_desc-brain_mask.nii.gz",
    ]
    for name in func_files:
        assert (func / name).is_file(), f"Missing functional file: {name}"

    # -- QC derivative (at least one regressor) --
    qc_files = list(func.glob(f"{bold_stem}_space-*_desc-xcp_*_quality.tsv"))
    assert qc_files, "No QC quality TSV files found"

    # -- Metrics derivatives --
    timeseries = list(func.glob(f"{bold_stem}_space-*_*_timeseries.tsv"))
    assert timeseries, "No timeseries TSV files found"

    correlations = list(func.glob(f"{bold_stem}_space-*_*_correlations.tsv"))
    assert correlations, "No correlation matrix TSV files found"
