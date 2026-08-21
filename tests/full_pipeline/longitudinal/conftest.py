"""Shared fixtures for longitudinal full-pipeline tests.

Mirrors the subprocess-driven style in ``tests/integration/test_all.py``:
session-scoped fixtures run each ``rbc`` invocation once and return the
output directory.  The ds000114 dataset is used (sub-01, ses-test +
ses-retest).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOWNLOAD_SCRIPT = _REPO_ROOT / "scripts" / "download_ds000114.py"
_DATASET_DIR = _REPO_ROOT / "tests" / "data" / "ds000114"
_DATASET_SENTINEL = (
    _DATASET_DIR / "sub-01" / "ses-test" / "anat" / "sub-01_ses-test_T1w.nii.gz"
)

_SUB = "01"
_TASK = "fingerfootlips"


def _rbc_exe() -> str:
    exe = shutil.which("rbc")
    assert exe is not None, "rbc CLI not found on PATH"
    return exe


def _run_rbc(
    args: Sequence[str], *, timeout: int = 7200
) -> subprocess.CompletedProcess[str]:
    """Run the ``rbc`` CLI and assert it exits cleanly."""
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


def _relative_files(root: Path) -> set[str]:
    """Return the set of file paths relative to *root*."""
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------------------
# Dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ds000114_dataset() -> Path:
    """Return the ds000114 BIDS dataset root, downloading it on first use."""
    if _DATASET_SENTINEL.exists():
        return _DATASET_DIR

    if not _DOWNLOAD_SCRIPT.exists():
        pytest.skip(f"download script missing: {_DOWNLOAD_SCRIPT}")

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not found on PATH; cannot run download script")
    # Explicit narrowing: pre-commit's mypy can't see pytest.skip as NoReturn
    # (pytest is unresolvable there), so it wouldn't narrow `uv` on its own.
    assert uv is not None

    result = subprocess.run(  # noqa: S603
        [uv, "run", str(_DOWNLOAD_SCRIPT), str(_DATASET_DIR)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not _DATASET_SENTINEL.exists():
        pytest.skip(
            "ds000114 download failed; skipping longitudinal tests.\n"
            f"--- stdout ---\n{result.stdout[-1000:]}\n"
            f"--- stderr ---\n{result.stderr[-1000:]}"
        )
    return _DATASET_DIR


@pytest.fixture(scope="session")
def _runner(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--runner")


# ---------------------------------------------------------------------------
# Stage-by-stage pipeline fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def longitudinal_pipeline_data(
    ds000114_dataset: Path,
    tmp_path_factory: pytest.TempPathFactory,
    _runner: str,
) -> Path:
    """Run the full longitudinal pipeline stage-by-stage.

    Executes: anatomical -> functional -> longitudinal template ->
    longitudinal anatomical -> longitudinal functional ->
    longitudinal metrics -> longitudinal qc.

    Both sessions (ses-test and ses-retest) are processed so the QC report
    summarizes the subject across sessions. Session-scoped so the cost is
    paid once across all tests. Returns the derivatives directory.
    """
    out = tmp_path_factory.mktemp("long_pipeline") / "derivatives"
    out.mkdir()

    raw = str(ds000114_dataset)
    deriv = str(out)
    common = [
        "--runner",
        _runner,
        "--participant-label",
        _SUB,
    ]

    # Cross-sectional anatomical
    _run_rbc(["anatomical", raw, "-o", deriv, *common])

    # Cross-sectional functional (all sessions)
    _run_rbc(
        [
            "functional",
            raw,
            deriv,
            "-o",
            deriv,
            *common,
            "--task",
            _TASK,
        ]
    )

    # Longitudinal template
    _run_rbc(
        [
            "longitudinal",
            "template",
            deriv,
            "-o",
            deriv,
            *common,
        ]
    )

    # Longitudinal anatomical
    _run_rbc(
        [
            "longitudinal",
            "anatomical",
            deriv,
            "-o",
            deriv,
            *common,
        ]
    )

    # Longitudinal functional
    _run_rbc(
        [
            "longitudinal",
            "functional",
            deriv,
            "-o",
            deriv,
            *common,
            "--task",
            _TASK,
        ]
    )

    # Longitudinal metrics
    _run_rbc(
        [
            "longitudinal",
            "metrics",
            deriv,
            "-o",
            deriv,
            *common,
            "--task",
            _TASK,
        ]
    )

    # Longitudinal QC
    _run_rbc(
        [
            "longitudinal",
            "qc",
            deriv,
            "-o",
            deriv,
            *common,
        ]
    )

    return out


@pytest.fixture(scope="session")
def longitudinal_all_data(
    ds000114_dataset: Path,
    tmp_path_factory: pytest.TempPathFactory,
    _runner: str,
) -> Path:
    """Run ``rbc longitudinal all`` and return the derivatives directory.

    Cross-sectional anatomical + functional must run first (``all`` does
    not re-run cross-sectional stages).
    """
    out = tmp_path_factory.mktemp("long_all") / "derivatives"
    out.mkdir()

    raw = str(ds000114_dataset)
    deriv = str(out)
    common = [
        "--runner",
        _runner,
        "--participant-label",
        _SUB,
    ]

    # Cross-sectional anatomical
    _run_rbc(["anatomical", raw, "-o", deriv, *common])

    # Cross-sectional functional (all sessions)
    _run_rbc(
        [
            "functional",
            raw,
            deriv,
            "-o",
            deriv,
            *common,
            "--task",
            _TASK,
        ]
    )

    # Full longitudinal pipeline (all sessions, matching sequential run)
    _run_rbc(
        [
            "longitudinal",
            "all",
            deriv,
            "-o",
            deriv,
            *common,
            "--task",
            _TASK,
        ]
    )

    return out
