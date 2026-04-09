"""Integration test for ``rbc all`` pipeline stage handoff.

Runs the full pipeline via ``rbc all`` (in-memory handoff) and via the four
individual subcommands in sequence (disk round-trip), then verifies that both
approaches complete without errors, produce key derivative files, and yield
identical outputs.

The BOLD timeseries is truncated to a small number of volumes so that
functional processing finishes in minutes rather than tens of minutes.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

_ORIG_DATASET = Path(__file__).parents[1] / "data" / "ds000001"

# Subject with no session in ds000001.
_SUB = "01"
_TASK = "balloonanalogrisktask"
_RUN = "01"

# Number of BOLD volumes to keep. Must be enough for nuisance regression
# (after discarding --start-tr 2) but small enough to be fast.
_BOLD_VOLUMES = 25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rbc_exe() -> str:
    exe = shutil.which("rbc")
    assert exe is not None, "rbc CLI not found on PATH"
    return exe


def _run_rbc(
    args: Sequence[str], *, timeout: int = 7200
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603
        [_rbc_exe(), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"rbc {args[0]} exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    return result


_COMMON_ARGS: list[str] = ["--participant-label", _SUB]


def _relative_files(root: Path) -> set[str]:
    """Return the set of file paths relative to *root*."""
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Fixtures — each pipeline variant runs once per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _runner(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--runner")


@pytest.fixture(scope="session")
def truncated_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Copy ds000001 and truncate the BOLD to *_BOLD_VOLUMES* volumes."""
    dest = tmp_path_factory.mktemp("ds") / "ds000001"
    shutil.copytree(_ORIG_DATASET, dest)

    bold = (
        dest
        / f"sub-{_SUB}"
        / "func"
        / f"sub-{_SUB}_task-{_TASK}_run-{_RUN}_bold.nii.gz"
    )
    img = nib.nifti1.load(bold)
    data = np.asarray(img.dataobj[:, :, :, :_BOLD_VOLUMES])
    nib.nifti1.save(nib.Nifti1Image(data, img.affine, img.header), bold)
    return dest


@pytest.fixture(scope="session")
def all_output(
    tmp_path_factory: pytest.TempPathFactory,
    _runner: str,
    truncated_dataset: Path,
) -> Path:
    """Run ``rbc all`` once and return the output directory."""
    out = tmp_path_factory.mktemp("all") / "derivatives"
    out.mkdir()
    _run_rbc(
        [
            "all",
            str(truncated_dataset),
            "-o",
            str(out),
            "--runner",
            _runner,
            *_COMMON_ARGS,
        ]
    )
    return out


@pytest.fixture(scope="session")
def sequential_output(
    tmp_path_factory: pytest.TempPathFactory,
    _runner: str,
    truncated_dataset: Path,
) -> Path:
    """Run the four individual subcommands in order and return the output dir."""
    out = tmp_path_factory.mktemp("sequential") / "derivatives"
    out.mkdir()

    runner_args = ["--runner", _runner, *_COMMON_ARGS]
    raw = str(truncated_dataset)
    deriv = str(out)

    # 1. anatomical (raw BIDS only)
    _run_rbc(["anatomical", raw, "-o", deriv, *runner_args])

    # 2. functional (raw BIDS + anat derivatives)
    _run_rbc(["functional", raw, deriv, "-o", deriv, *runner_args])

    # 3. metrics (raw BIDS + derivatives from previous stages)
    _run_rbc(["metrics", raw, deriv, "-o", deriv, *runner_args])

    # 4. qc (raw BIDS + derivatives from previous stages)
    _run_rbc(["qc", raw, deriv, "-o", deriv, *runner_args])

    return out


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _assert_derivatives_exist(output_dir: Path) -> None:
    """Check that all expected derivative files are present."""
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

    # -- QC --
    qc_files = list(func.glob(f"{bold_stem}_space-*_desc-xcp_*_quality.tsv"))
    assert qc_files, "No QC quality TSV files found"

    # -- Metrics --
    assert list(func.glob(f"{bold_stem}_space-*_*_timeseries.tsv")), (
        "No timeseries TSV files found"
    )
    assert list(func.glob(f"{bold_stem}_space-*_*_correlations.tsv")), (
        "No correlation matrix TSV files found"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rbc_all_produces_derivatives(all_output: Path) -> None:
    """``rbc all`` runs end-to-end and writes expected derivative files."""
    _assert_derivatives_exist(all_output)


@pytest.mark.slow
def test_sequential_produces_derivatives(sequential_output: Path) -> None:
    """Running anatomical/functional/metrics/qc individually produces the same files."""
    _assert_derivatives_exist(sequential_output)


@pytest.mark.slow
def test_all_vs_sequential_outputs_match(
    all_output: Path,
    sequential_output: Path,
) -> None:
    """Both invocation styles must produce the same set of derivative files."""
    all_files = _relative_files(all_output)
    seq_files = _relative_files(sequential_output)

    missing_from_seq = all_files - seq_files
    extra_in_seq = seq_files - all_files

    assert not missing_from_seq, (
        "Files produced by 'rbc all' but missing from sequential run:\n"
        + "\n".join(sorted(missing_from_seq))
    )
    assert not extra_in_seq, (
        "Files produced by sequential run but missing from 'rbc all':\n"
        + "\n".join(sorted(extra_in_seq))
    )
