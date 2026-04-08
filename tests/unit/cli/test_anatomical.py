"""Unit tests for Anatomical CLI module."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rbc.cli import anatomical

if TYPE_CHECKING:
    from pathlib import Path


class TestAnatomicalArgs:
    """Tests for AnatomicalArgs validation."""

    def test_parser_from_namespace(self, tmp_path: Path) -> None:
        """Tests parser successfully validates namespace."""
        input_dir = tmp_path / "input"
        input_dir.touch()
        ns = argparse.Namespace(
            runner="local",
            verbose=False,
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            participant_label=[],
            session_label=[],
            tmp_dir=None,
        )
        args = anatomical.AnatomicalArgs.validate_namespace(ns)
        assert isinstance(args, anatomical.AnatomicalArgs)
