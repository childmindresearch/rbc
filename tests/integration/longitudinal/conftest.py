"""Fixtures for longitudinal integration tests.

These fixtures own the ds000114 test dataset lifecycle (download + reuse)
and run the cross-sectional anatomical stage once so that each
longitudinal test can build on ``desc-brain`` T1w derivatives without
paying the preprocessing cost per test.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOWNLOAD_SCRIPT = _REPO_ROOT / "scripts" / "download_ds000114.py"
_DATASET_DIR = _REPO_ROOT / "tests" / "data" / "ds000114"
_DATASET_SENTINEL = (
    _DATASET_DIR / "sub-01" / "ses-test" / "anat" / "sub-01_ses-test_T1w.nii.gz"
)


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
def runner_backend(request: pytest.FixtureRequest) -> str:
    """Styx runner backend selected via ``--runner`` on the pytest CLI."""
    return request.config.getoption("--runner")


@pytest.fixture(scope="session")
def ds000114_anat_derivatives(
    ds000114_dataset: Path,
    tmp_path_factory: pytest.TempPathFactory,
    runner_backend: str,
) -> Path:
    """Run ``rbc anatomical`` once against ds000114 sub-01 (both sessions).

    The longitudinal template stage consumes the cross-sectional
    ``desc-brain`` T1w derivatives, so we produce them up front. Session
    scope ensures we only pay the registration + brain extraction cost
    once across all longitudinal integration tests.
    """
    rbc = shutil.which("rbc")
    if rbc is None:
        pytest.skip("rbc CLI not found on PATH")

    out = tmp_path_factory.mktemp("ds000114_derivatives")
    result = subprocess.run(  # noqa: S603
        [
            rbc,
            "anatomical",
            str(ds000114_dataset),
            "-o",
            str(out),
            "--runner",
            runner_backend,
            "--participant-label",
            "01",
        ],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert result.returncode == 0, (
        f"rbc anatomical exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    return out
