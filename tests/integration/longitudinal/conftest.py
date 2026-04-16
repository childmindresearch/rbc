"""Fixtures for longitudinal integration tests.

Mirrors the subprocess-driven style in ``tests/integration/test_all.py``:
session-scoped fixtures run each ``rbc`` invocation once and return the
output directory; tests just assert on the resulting BIDS tree.
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
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return result


@pytest.fixture(scope="session")
def ds000114_dataset() -> Path:
    """Return the ds000114 BIDS dataset root, downloading it on first use.

    Skips the calling test if the dataset can't be fetched (most commonly
    because the S3 mirror is unreachable or the uv runner isn't available),
    so local developers without network access don't see hard failures
    while CI still exercises the path.
    """
    if _DATASET_SENTINEL.exists():
        return _DATASET_DIR

    if not _DOWNLOAD_SCRIPT.exists():
        pytest.skip(f"download script missing: {_DOWNLOAD_SCRIPT}")

    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not found on PATH; cannot run download script")

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


@pytest.fixture(scope="session")
def ds000114_anat_derivatives(
    ds000114_dataset: Path,
    tmp_path_factory: pytest.TempPathFactory,
    _runner: str,
) -> Path:
    """Run ``rbc anatomical`` once against ds000114 sub-01 (both sessions).

    The longitudinal template stage consumes the cross-sectional
    ``desc-brain`` T1w derivatives, so we produce them up front. Session
    scope ensures we only pay the registration + brain extraction cost
    once across all longitudinal integration tests.
    """
    out = tmp_path_factory.mktemp("ds000114_derivatives")
    _run_rbc(
        [
            "anatomical",
            str(ds000114_dataset),
            "-o",
            str(out),
            "--runner",
            _runner,
            "--participant-label",
            _SUB,
        ],
    )
    return out


@pytest.fixture(scope="session")
def longitudinal_template_output(
    ds000114_anat_derivatives: Path,
    _runner: str,
) -> Path:
    """Run ``rbc longitudinal template`` once and return the derivatives dir.

    Writes the ``ses-longitudinal`` tree alongside the per-session
    cross-sectional anat outputs, the same layout downstream longitudinal
    stages will consume.
    """
    _run_rbc(
        [
            "longitudinal",
            "template",
            str(ds000114_anat_derivatives),
            "-o",
            str(ds000114_anat_derivatives),
            "--runner",
            _runner,
            "--participant-label",
            _SUB,
        ],
    )
    return ds000114_anat_derivatives


@pytest.fixture(scope="session")
def ds000114_func_derivatives(
    ds000114_dataset: Path,
    ds000114_anat_derivatives: Path,
    _runner: str,
) -> Path:
    """Run ``rbc functional`` on ds000114 sub-01 ses-test.

    Produces cross-sectional functional derivatives (including raw
    regressor ``.1D`` files) that the longitudinal functional stage
    consumes.  Writes into the same derivatives tree as the anatomical
    stage so all outputs are visible to downstream fixtures.

    Only ses-test is processed (one session is sufficient to exercise
    the longitudinal functional chain).
    """
    # Note: do NOT pass --task here.  The Filters.apply() task filter
    # applies to ALL rows including anat, and anat rows have task=null,
    # so --task would drop all anat rows and break resolve_functional.
    _run_rbc(
        [
            "functional",
            str(ds000114_dataset),
            str(ds000114_anat_derivatives),
            "-o",
            str(ds000114_anat_derivatives),
            "--runner",
            _runner,
            "--participant-label",
            _SUB,
            "--session-label",
            "test",
        ],
    )
    return ds000114_anat_derivatives


@pytest.fixture(scope="session")
def longitudinal_func_output(
    ds000114_func_derivatives: Path,
    longitudinal_template_output: Path,  # noqa: ARG001 — fixture dep for ordering
    _runner: str,
) -> Path:
    """Run ``rbc longitudinal functional`` on ds000114 sub-01 ses-test.

    Produces longitudinal functional derivatives (warped BOLD,
    per-regressor regressed/cleaned BOLD) by consuming the
    cross-sectional functional outputs and the longitudinal template.
    Both ``ds000114_func_derivatives`` and ``longitudinal_template_output``
    write into the same derivatives directory, so all files are visible.
    """
    _run_rbc(
        [
            "longitudinal",
            "functional",
            str(ds000114_func_derivatives),
            "-o",
            str(ds000114_func_derivatives),
            "--runner",
            _runner,
            "--participant-label",
            _SUB,
            "--session-label",
            "test",
        ],
    )
    return ds000114_func_derivatives
