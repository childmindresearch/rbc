"""Unit tests for QC CLI module."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rbc.cli.main import cli, create_parser
from rbc.cli.qc import QCArgs


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base QC argument namespace with sensible defaults."""
    input_dir = tmp_path / "input"
    input_dir.touch()
    output_dir = tmp_path / "output"
    return argparse.Namespace(
        runner="local",
        verbose=False,
        input_dirs=[input_dir],
        output_dir=output_dir,
        participant_label=[],
        session_label=[],
        task=None,
        start_tr=2,
        regressor=["36-parameter"],
        tmp_dir=None,
    )


class TestQCArgs:
    """Tests for QCArgs validation.

    Covers default values, custom values, preservation of all fields
    through validate_namespace, and input validation for task and start_tr.
    """

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates a well-formed namespace."""
        args = QCArgs.validate_namespace(base_args)
        assert isinstance(args, QCArgs)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Default values for task, start_tr, regressor, and labels are preserved."""
        args = QCArgs.validate_namespace(base_args)
        assert args.task is None
        assert args.start_tr == 2
        assert args.regressor == ["36-parameter"]
        assert args.participant_label == []
        assert args.session_label == []

    def test_custom_start_tr(self, base_args: argparse.Namespace) -> None:
        """Custom start_tr is correctly preserved through validation."""
        base_args.start_tr = 5
        args = QCArgs.validate_namespace(base_args)
        assert args.start_tr == 5

    def test_invalid_start_tr_zero(self, base_args: argparse.Namespace) -> None:
        """start_tr of 0 raises ValueError."""
        base_args.start_tr = 0
        with pytest.raises(ValueError, match="Start TR should be greater than 0"):
            QCArgs.validate_namespace(base_args)

    def test_invalid_start_tr_negative(self, base_args: argparse.Namespace) -> None:
        """Negative start_tr raises ValueError."""
        base_args.start_tr = -1
        with pytest.raises(ValueError, match="Start TR should be greater than 0"):
            QCArgs.validate_namespace(base_args)

    @pytest.mark.parametrize("regressor", ["36-parameter", "aCompCor"])
    def test_valid_regressors(
        self, base_args: argparse.Namespace, regressor: str
    ) -> None:
        """Both supported regressor options pass validation."""
        base_args.regressor = [regressor]
        args = QCArgs.validate_namespace(base_args)
        assert args.regressor == [regressor]

    def test_task_preserved(self, base_args: argparse.Namespace) -> None:
        """Provided task label is preserved through validation."""
        base_args.task = "rest"
        args = QCArgs.validate_namespace(base_args)
        assert args.task == "rest"

    @pytest.mark.parametrize(
        "task",
        ["rest", "nback", "faces+n+back", "task123", None],
        ids=["simple", "alphanumeric", "plus_separator", "with_digits", "none"],
    )
    def test_valid_task_labels(
        self, base_args: argparse.Namespace, task: str | None
    ) -> None:
        """Valid task labels pass validation."""
        base_args.task = task
        args = QCArgs.validate_namespace(base_args)
        assert args.task == task

    @pytest.mark.parametrize(
        "task",
        ["faces n-back", "task label", "task!", "task/name"],
        ids=["space_hyphen", "space", "special_char", "slash"],
    )
    def test_invalid_task_labels(
        self, base_args: argparse.Namespace, task: str
    ) -> None:
        """Invalid task labels raise ValueError."""
        base_args.task = task
        with pytest.raises(ValueError, match="Task must contain only alphanumeric"):
            QCArgs.validate_namespace(base_args)


class TestQCRegistration:
    """Test that QC command is registered and discoverable."""

    def test_qc_command_registered(self) -> None:
        """Test QC workflow is available in parser."""
        result = cli(["qc", "/input", "-o", "/output", "--help"])
        assert result == 0

    def test_qc_parser_has_task(self) -> None:
        """Test QC subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["qc", "/input", "-o", "/output", "--task", "rest"])
        assert args.task == "rest"

    def test_qc_parser_has_start_tr(self) -> None:
        """Test QC subparser includes --start-tr argument."""
        parser = create_parser()
        args = parser.parse_args(["qc", "/input", "-o", "/output", "--start-tr", "5"])
        assert args.start_tr == 5

    def test_qc_parser_has_regressor(self) -> None:
        """Test QC subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["qc", "/input", "-o", "/output"])
        assert args.regressor == ["36-parameter"]

    def test_qc_parser_task_default_none(self) -> None:
        """Test QC subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["qc", "/input", "-o", "/output"])
        assert args.task is None

    def test_qc_parser_start_tr_default(self) -> None:
        """Test QC subparser --start-tr defaults to 2."""
        parser = create_parser()
        args = parser.parse_args(["qc", "/input", "-o", "/output"])
        assert args.start_tr == 2
