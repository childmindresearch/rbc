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


def _brain_row(sub: str, ses: str) -> tuple:
    path = f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_desc-brain_T1w.nii.gz"
    return (
        "anat",
        "T1w",
        ".nii.gz",
        sub,
        ses,
        None,
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
        result = discover_template_inputs(df)
        assert {ti.sub for ti in result} == {"01", "02"}
        for ti in result:
            assert sorted(ti.sessions) == ["baseline", "vis2"]
            assert all(p.suffix == ".gz" for p in ti.files)

    def test_skips_single_session_subject(self) -> None:
        """Per-subject volume check must skip single-session subjects (#19)."""
        df = _df(
            _brain_row("01", "baseline"),
            _brain_row("01", "vis2"),
            _brain_row("02", "baseline"),
        )
        result = discover_template_inputs(df)
        assert [ti.sub for ti in result] == ["01"]

    def test_excludes_existing_longitudinal(self) -> None:
        """Pre-existing longitudinal templates are not re-included as inputs."""
        df = _df(
            _brain_row("01", "baseline"),
            _brain_row("01", "vis2"),
            _brain_row("01", "longitudinal"),
        )
        result = discover_template_inputs(df)
        assert len(result) == 1
        assert sorted(result[0].sessions) == ["baseline", "vis2"]

    def test_empty_when_no_subjects(self) -> None:
        """Empty input yields empty output."""
        empty = pl.DataFrame({c: [] for c in _SCHEMA})
        assert discover_template_inputs(empty) == []


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
