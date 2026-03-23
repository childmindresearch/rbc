"""Unit tests for rbc.context and rbc.core.bids.Bids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from rbc.context import PipelineContext
from rbc.core.bids import Bids, bids_safe_label

if TYPE_CHECKING:
    from pathlib import Path


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


class TestBids:
    """Tests for the Bids builder."""

    @pytest.fixture
    def pipe_ctx(self, tmp_path: Path) -> PipelineContext:
        """Create a PipelineContext for testing."""
        return PipelineContext(sub="01", ses="baseline", output_dir=tmp_path)

    @pytest.fixture
    def src_file(self, tmp_path: Path) -> Path:
        """Create a dummy source file."""
        src = tmp_path / "input.nii.gz"
        src.write_bytes(b"\x00")
        return src

    def test_bids_factory_returns_bids(self, pipe_ctx: PipelineContext) -> None:
        """Verify bids() factory returns a Bids instance."""
        b = pipe_ctx.bids(datatype="anat")
        assert isinstance(b, Bids)

    def test_save_copies_file(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify .save() copies source to BIDS-named path."""
        func = pipe_ctx.bids(datatype="func", entities={"task": "rest", "run": 1})
        result = func.save(src_file, suffix="bold", desc="preproc")
        assert result.exists()
        assert "task-rest" in result.name
        assert "run-1" in result.name
        assert "desc-preproc" in result.name

    def test_derive_inherits_entities(
        self, pipe_ctx: PipelineContext, src_file: Path
    ) -> None:
        """Verify derive() carries forward parent entities."""
        func = pipe_ctx.bids(datatype="func", entities={"task": "rest"})
        mni = func.derive(space="MNI152")
        result = mni.save(src_file, suffix="bold")
        assert "task-rest" in result.name
        assert "space-MNI152" in result.name

    def test_derive_overrides(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify derive() overrides parent values."""
        func = pipe_ctx.bids(datatype="func", space="native")
        mni = func.derive(space="MNI152")
        result = mni.save(src_file, suffix="bold")
        assert "space-MNI152" in result.name
        assert "space-native" not in result.name

    def test_save_overrides(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify per-call overrides on save()."""
        func = pipe_ctx.bids(datatype="func")
        result = func.save(src_file, suffix="bold", space="MNI152", desc="preproc")
        assert "space-MNI152" in result.name
        assert "desc-preproc" in result.name

    def test_extra_merges_on_derive(
        self, pipe_ctx: PipelineContext, src_file: Path
    ) -> None:
        """Verify extra dicts merge on derive()."""
        func = pipe_ctx.bids(datatype="func", extra={"reg": "36parameter"})
        derived = func.derive(extra={"mode": "image"})
        result = derived.save(src_file, suffix="bold")
        assert "reg-36parameter" in result.name
        assert "mode-image" in result.name

    def test_extra_merges_on_save(
        self, pipe_ctx: PipelineContext, src_file: Path
    ) -> None:
        """Verify per-call extra merges with session extra."""
        func = pipe_ctx.bids(datatype="func", extra={"reg": "36parameter"})
        result = func.save(src_file, suffix="bold", extra={"mode": "image"})
        assert "reg-36parameter" in result.name
        assert "mode-image" in result.name

    def test_entity_ordering(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify BIDS entity ordering in output filename."""
        func = pipe_ctx.bids(
            datatype="func",
            entities={"task": "rest", "acq": "1400", "rec": "magnitude", "dir": "AP"},
        )
        result = func.save(src_file, suffix="bold")
        name = result.name
        assert name.index("acq-") < name.index("rec-") < name.index("dir-")

    def test_save_dir(self, pipe_ctx: PipelineContext, tmp_path: Path) -> None:
        """Verify .save_dir() copies directory."""
        src_dir = tmp_path / "src_dir"
        src_dir.mkdir()
        (src_dir / "dummy.txt").write_text("test")

        func = pipe_ctx.bids(datatype="func", entities={"task": "rest", "run": 1})
        result = func.save_dir(src_dir, suffix="motion")
        assert result.exists()
        assert result.is_dir()
        assert "task-rest" in result.name

    def test_no_entities(self, pipe_ctx: PipelineContext, src_file: Path) -> None:
        """Verify save works with no entities (just sub/ses/datatype)."""
        anat = pipe_ctx.bids(datatype="anat")
        result = anat.save(src_file, suffix="T1w", desc="brain")
        assert "sub-01" in result.name
        assert "ses-baseline" in result.name
        assert "desc-brain" in result.name
