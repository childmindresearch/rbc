"""Unit tests for rbc.context."""

from __future__ import annotations

from pathlib import Path

import pytest

from rbc.context import PipelineContext
from rbc.core.bids import bids_safe_label


class TestBidsSafeLabel:
    """Tests for bids_safe_label."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("36-parameter", "36parameter"),
            ("aCompCor", "aCompCor"),
            ("preproc", "preproc"),
            ("hello world", "helloworld"),
            ("a_b-c+d", "abc+d"),
            ("MNI152", "MNI152"),
        ],
    )
    def test_strips_invalid_chars(self, raw: str, expected: str) -> None:
        """Verify that non-BIDS characters are removed."""
        assert bids_safe_label(raw) == expected


class TestExportBidsEntities:
    """Tests for PipelineContext.export() with acq/dir/rec/echo entities."""

    @pytest.fixture()
    def pipe_ctx(self, tmp_path: Path) -> PipelineContext:
        """Create a PipelineContext for testing."""
        return PipelineContext(sub="01", ses="baseline", output_dir=tmp_path)

    @pytest.fixture()
    def src_file(self, tmp_path: Path) -> Path:
        """Create a dummy source file."""
        src = tmp_path / "input.nii.gz"
        src.write_bytes(b"\x00")
        return src

    def test_acq_in_filename(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify acq entity appears in output filename."""
        result = pipe_ctx.export(
            src_file,
            datatype="func",
            suffix="bold",
            task="rest",
            acq="1400",
        )
        assert "acq-1400" in result.name

    def test_dir_in_filename(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify dir entity appears in output filename."""
        result = pipe_ctx.export(
            src_file,
            datatype="func",
            suffix="bold",
            task="rest",
            dir="AP",
        )
        assert "dir-AP" in result.name

    def test_rec_in_filename(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify rec entity appears in output filename."""
        result = pipe_ctx.export(
            src_file,
            datatype="anat",
            suffix="T1w",
            rec="magnitude",
        )
        assert "rec-magnitude" in result.name

    def test_echo_in_filename(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify echo entity appears in output filename."""
        result = pipe_ctx.export(
            src_file,
            datatype="func",
            suffix="bold",
            task="rest",
            echo=1,
        )
        assert "echo-1" in result.name

    def test_multiple_entities(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify multiple new entities appear together."""
        result = pipe_ctx.export(
            src_file,
            datatype="func",
            suffix="bold",
            task="rest",
            acq="1400",
            dir="AP",
            rec="magnitude",
            echo=2,
            run=1,
        )
        name = result.name
        assert "acq-1400" in name
        assert "dir-AP" in name
        assert "rec-magnitude" in name
        assert "echo-2" in name
        assert "run-1" in name

    def test_entity_ordering(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify BIDS entity ordering: acq before rec before dir before run."""
        result = pipe_ctx.export(
            src_file,
            datatype="func",
            suffix="bold",
            task="rest",
            acq="1400",
            rec="magnitude",
            dir="AP",
            run=1,
        )
        name = result.name
        acq_pos = name.index("acq-")
        rec_pos = name.index("rec-")
        dir_pos = name.index("dir-")
        run_pos = name.index("run-")
        assert acq_pos < rec_pos < dir_pos < run_pos

    def test_none_values_backward_compat(
        self, pipe_ctx: PipelineContext, src_file: Path
    ) -> None:
        """Verify None values produce same output as before (no extra entities)."""
        result = pipe_ctx.export(
            src_file,
            datatype="func",
            suffix="bold",
            task="rest",
            acq=None,
            rec=None,
            dir=None,
            echo=None,
        )
        name = result.name
        assert "acq-" not in name
        assert "rec-" not in name
        assert "dir-" not in name
        assert "echo-" not in name

    def test_export_dir_with_entities(
        self, pipe_ctx: PipelineContext, tmp_path: Path
    ) -> None:
        """Verify export_dir also supports the new entities."""
        src_dir = tmp_path / "src_dir"
        src_dir.mkdir()
        (src_dir / "dummy.txt").write_text("test")

        result = pipe_ctx.export_dir(
            src_dir,
            datatype="func",
            suffix="bold",
            task="rest",
            acq="1400",
            dir="AP",
        )
        name = result.name
        assert "acq-1400" in name
        assert "dir-AP" in name
