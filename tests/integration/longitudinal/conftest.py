"""Fixtures for longitudinal integration tests.

Mirrors the subprocess-driven style in ``tests/integration/test_all.py``:
session-scoped fixtures run each ``rbc`` invocation once and return the
output directory; tests just assert on the resulting BIDS tree.
"""

from __future__ import annotations

import os
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
    args: Sequence[str], *, timeout: int = 7200, bypass_cache: bool = False
) -> subprocess.CompletedProcess[str]:
    """Invoke ``rbc`` with ``-vv`` so container output reaches the subprocess.

    Without ``-vv`` the styx runner logger stays at WARNING and
    container-side output never makes it into the captured stderr, which
    leaves the test with only a bare ``returncode=1`` to diagnose from.

    Set ``bypass_cache=True`` to run with ``RBC_STYXCACHE_DIR`` cleared.
    styxcache's tee handlers swallow container stderr into an internal
    gzip that's discarded on exception, so a failing niwrap call under
    cache leaves no diagnostic trail; bypassing cache restores raw
    podman stderr passthrough.
    """
    env = os.environ.copy()
    if bypass_cache:
        env.pop("RBC_STYXCACHE_DIR", None)

    result = subprocess.run(  # noqa: S603
        [_rbc_exe(), *args, "-vv"],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
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
        # Template generation is still returning silent non-zero on the
        # nightly; cache-bypass here gives us raw container stderr on
        # failure without losing the anat fixture's cache benefit.
        bypass_cache=True,
    )
    return ds000114_anat_derivatives
