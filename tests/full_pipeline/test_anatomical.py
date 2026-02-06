"""Full e2e tests for different pipelines."""

import pathlib as pl
from types import SimpleNamespace

from rbc.workflows.anatomical import single_session


def test_single_session(test_subject: SimpleNamespace, tmp_path: pl.Path) -> None:
    """e2e test for single session anatomical workflow."""
    subject_id = f"sub-{test_subject.id}"
    expected_output_dir = tmp_path / subject_id / "anat"
    expected_fnames = [
        "desc-T1w_mask.nii.gz",
        "desc-brain_T1w.nii.gz",
        "desc-csf_mask.nii.gz",
        "desc-gm_mask.nii.gz",
        "desc-wm_mask.nii.gz",
        "from-T1w_to-template_mode-image_xfm.nii.gz",
        "from-template_to-T1w_mode-image_xfm.nii.gz",
    ]
    single_session(test_subject.t1w, output_dir=tmp_path)

    assert expected_output_dir.exists()
    assert all(
        (expected_output_dir / f"{subject_id}_{fname}").exists()
        for fname in expected_fnames
    )
