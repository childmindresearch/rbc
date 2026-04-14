"""Unit tests for ``rbc.bids.longitudinal.template``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from rbc.bids.longitudinal.template import (
    discover_template_inputs,
    export_template,
)
from rbc.context import RunContext
from rbc.workflows.longitudinal.template import LongitudinalTemplateOutputs

if TYPE_CHECKING:
    from pathlib import Path

_SCHEMA = [
    "datatype",
    "suffix",
    "ext",
    "sub",
    "ses",
    "space",
    "task",
    "run",
    "desc",
    "root",
    "path",
]


def _df(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(dict(zip(_SCHEMA, zip(*rows, strict=True), strict=True)))


def _brain_row(sub: str, ses: str, space: str | None = None) -> tuple:
    space_part = f"_space-{space}" if space else ""
    path = (
        f"sub-{sub}/ses-{ses}/anat/"
        f"sub-{sub}_ses-{ses}{space_part}_desc-brain_T1w.nii.gz"
    )
    return (
        "anat",
        "T1w",
        ".nii.gz",
        sub,
        ses,
        space,
        None,
        None,
        "brain",
        "/data",
        path,
    )


class TestDiscoverTemplateInputs:
    """Tests for :func:`discover_template_inputs`."""

    def test_groups_by_subject(self) -> None:
        """Multi-session subjects yield one TemplateInputs each."""
        df = _df(
            _brain_row("01", "baseline"),
            _brain_row("01", "vis2"),
            _brain_row("02", "baseline"),
            _brain_row("02", "vis2"),
        )
        inputs, skipped = discover_template_inputs(df)
        assert {ti.sub for ti in inputs} == {"01", "02"}
        assert skipped == []
        for ti in inputs:
            assert sorted(ti.sessions) == ["baseline", "vis2"]
            assert all(p.suffix == ".gz" for p in ti.files)

    def test_reports_single_session_subject(self) -> None:
        """Single-session subjects are reported separately (bug #19)."""
        df = _df(
            _brain_row("01", "baseline"),
            _brain_row("01", "vis2"),
            _brain_row("02", "baseline"),
        )
        inputs, skipped = discover_template_inputs(df)
        assert [ti.sub for ti in inputs] == ["01"]
        assert skipped == ["02"]

    def test_excludes_existing_longitudinal(self) -> None:
        """Pre-existing longitudinal templates are not re-included as inputs."""
        df = _df(
            _brain_row("01", "baseline"),
            _brain_row("01", "vis2"),
            _brain_row("01", "longitudinal"),
        )
        inputs, skipped = discover_template_inputs(df)
        assert len(inputs) == 1
        assert sorted(inputs[0].sessions) == ["baseline", "vis2"]
        assert skipped == []

    def test_empty_when_no_subjects(self) -> None:
        """Empty input yields empty output."""
        empty = pl.DataFrame({c: [] for c in _SCHEMA})
        assert discover_template_inputs(empty) == ([], [])

    def test_excludes_mni_registered_brains(self) -> None:
        """Cross-sectional MNI-registered desc-brain T1ws are not template inputs.

        Cross-sectional anat writes both a native-space and an MNI-registered
        ``desc-brain`` T1w per session. Without filtering on ``space.is_null()``
        the latter is picked up as a second input per session, producing
        duplicate LTA filenames in the mri_robust_template invocation.
        """
        df = _df(
            _brain_row("01", "test"),
            _brain_row("01", "test", space="MNI152NLin6Asym"),
            _brain_row("01", "retest"),
            _brain_row("01", "retest", space="MNI152NLin6Asym"),
        )
        inputs, skipped = discover_template_inputs(df)
        assert len(inputs) == 1
        assert sorted(inputs[0].sessions) == ["retest", "test"]
        assert len(inputs[0].files) == 2
        assert all("space-" not in str(p) for p in inputs[0].files)
        assert skipped == []


class TestExportTemplate:
    """Tests for :func:`export_template`."""

    def test_writes_template_and_xfms(self, tmp_path: Path) -> None:
        """Template + per-session xfms land at the expected BIDS paths."""
        template_src = tmp_path / "src_template.nii.gz"
        template_src.write_bytes(b"\x00")
        xfm_baseline = tmp_path / "xfm_baseline.txt"
        xfm_baseline.write_text("baseline")
        xfm_vis2 = tmp_path / "xfm_vis2.txt"
        xfm_vis2.write_text("vis2")

        outputs = LongitudinalTemplateOutputs(
            template=template_src,
            sessions=["baseline", "vis2"],
            transforms=[xfm_baseline, xfm_vis2],
        )

        out_dir = tmp_path / "out"
        ctx = RunContext(sub="01", ses=None, output_dir=out_dir)
        tpl = ctx.bids(datatype="anat").derive(ses="longitudinal")

        export_template(tpl, outputs)

        long_dir = out_dir / "sub-01" / "ses-longitudinal" / "anat"
        assert (long_dir / "sub-01_ses-longitudinal_T1w.nii.gz").exists()
        assert (
            long_dir
            / (
                "sub-01_ses-longitudinal_from-baseline_to-longitudinal"
                "_mode-image_xfm.txt"
            )
        ).exists()
        assert (
            long_dir
            / ("sub-01_ses-longitudinal_from-vis2_to-longitudinal_mode-image_xfm.txt")
        ).exists()
