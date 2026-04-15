"""Integration test for ``rbc longitudinal template``.

Deferred from Stage 2 of the longitudinal refactor (tracker #301,
Stage 2 landed in PR #306). Depends on the ds000114 multi-session
fixture and the cross-sectional anatomical pre-run that produces the
``desc-brain`` T1w derivatives the template stage consumes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.slow
def test_rbc_longitudinal_template_builds_bids_tree(
    longitudinal_template_output: Path,
) -> None:
    """The ``ses-longitudinal`` BIDS tree contains the expected files."""
    ses_long = longitudinal_template_output / "sub-01" / "ses-longitudinal" / "anat"
    expected = [
        "sub-01_ses-longitudinal_T1w.nii.gz",
        "sub-01_ses-longitudinal_from-test_to-longitudinal_mode-image_xfm.txt",
        "sub-01_ses-longitudinal_from-retest_to-longitudinal_mode-image_xfm.txt",
    ]
    missing = [name for name in expected if not (ses_long / name).is_file()]
    if missing:
        tree = sorted(
            str(p.relative_to(longitudinal_template_output))
            for p in longitudinal_template_output.rglob("*")
            if p.is_file()
        )
        pytest.fail(
            "Missing expected longitudinal derivatives:\n  "
            + "\n  ".join(missing)
            + "\n--- file tree ---\n"
            + "\n".join(tree)
        )
