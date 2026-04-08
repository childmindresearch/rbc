"""Unit tests for Longitudinal CLI module."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rbc.cli.longitudinal import LongitudinalArgs


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base argument namespace."""
    input_dir = tmp_path / "input"
    input_dir.touch()
    return argparse.Namespace(
        runner="local",
        verbose=False,
        input_dir=input_dir,
        output_dir=tmp_path / "output",
        participant_label=[],
        session_label=[],
        anatomical=True,
        functional=False,
        tmp_dir=None,
    )


class TestLongitudinalArgs:
    """Tests for LongitudinalArgs validation."""

    @pytest.mark.parametrize(
        ("anat", "func"), [(True, False), (False, True), (True, True)]
    )
    def test_valid_flag_combinations(
        self,
        base_args: argparse.Namespace,
        anat: bool,  # noqa: FBT001
        func: bool,  # noqa: FBT001
    ) -> None:
        """Test different combination of valid longitudinal flags."""
        base_args.anatomical, base_args.functional = anat, func
        args = LongitudinalArgs.validate_namespace(base_args)
        assert args.anatomical is anat
        assert args.functional is func

    def test_no_flags_raises(self, base_args: argparse.Namespace) -> None:
        """Test error raised if no processing selected."""
        base_args.anatomical = base_args.functional = False
        with pytest.raises(ValueError, match="At least one of"):
            LongitudinalArgs.validate_namespace(base_args)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Test defaults."""
        args = LongitudinalArgs.validate_namespace(base_args)
        assert args.participant_label == []
        assert args.session_label == []
