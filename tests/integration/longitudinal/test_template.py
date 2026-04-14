"""Integration test for ``rbc longitudinal template``.

Deferred from Stage 2 of the longitudinal refactor (tracker #301,
Stage 2 landed in PR #306). Depends on the ds000114 multi-session test
fixture; cross-sectional anatomical derivatives are produced by the
session-scoped ``ds000114_anat_derivatives`` fixture so the template
stage has ``desc-brain`` T1w volumes to consume.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
def test_rbc_longitudinal_template_builds_bids_tree(
    ds000114_anat_derivatives: Path,
    runner_backend: str,
) -> None:
    """Run ``rbc longitudinal template`` and verify the BIDS output tree."""
    rbc = shutil.which("rbc")
    assert rbc is not None, "rbc CLI not found on PATH"

    result = subprocess.run(  # noqa: S603
        [
            rbc,
            "longitudinal",
            "template",
            str(ds000114_anat_derivatives),
            "-o",
            str(ds000114_anat_derivatives),
            "--runner",
            runner_backend,
            "--participant-label",
            "01",
        ],
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert result.returncode == 0, (
        f"rbc longitudinal template exited with code {result.returncode}\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )

    ses_long = ds000114_anat_derivatives / "sub-01" / "ses-longitudinal" / "anat"
    expected = [
        "sub-01_ses-longitudinal_T1w.nii.gz",
        "sub-01_ses-longitudinal_from-test_to-longitudinal_mode-image_xfm.txt",
        "sub-01_ses-longitudinal_from-retest_to-longitudinal_mode-image_xfm.txt",
    ]
    missing = [name for name in expected if not (ses_long / name).is_file()]
    if missing:
        tree = sorted(
            str(p.relative_to(ds000114_anat_derivatives))
            for p in ds000114_anat_derivatives.rglob("*")
            if p.is_file()
        )
        pytest.fail(
            "Missing expected longitudinal derivatives:\n  "
            + "\n  ".join(missing)
            + "\n--- file tree ---\n"
            + "\n".join(tree)
        )
