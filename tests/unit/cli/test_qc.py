"""Unit tests for QC CLI module."""

import argparse
from pathlib import Path

import pytest

from rbc.cli.main import cli, create_parser
from rbc.cli.qc import QCArgs


class TestQCArgs:
    """Test suite for QCArgs validation."""

    @pytest.fixture
    def qc_namespace(self, tmp_path: Path) -> argparse.Namespace:
        """Fixture for QC argument namespace."""
        input_dir = tmp_path / "input"
        input_dir.touch()
        output_dir = tmp_path / "output"
        return argparse.Namespace(
            runner="local",
            verbose=0,
            input_dir=input_dir,
            output_dir=output_dir,
            participant_label=[],
            session_label=[],
            task=None,
            start_tr=2,
            regressor="36-parameter",
        )

    def test_validate_namespace(self, qc_namespace: argparse.Namespace) -> None:
        """Test QCArgs validates successfully with valid args."""
        args = QCArgs.validate_namespace(qc_namespace)
        assert isinstance(args, QCArgs)
        assert args.task is None
        assert args.start_tr == 2
        assert args.regressor == "36-parameter"

    def test_validate_with_task(self, qc_namespace: argparse.Namespace) -> None:
        """Test QCArgs preserves task filter."""
        qc_namespace.task = "rest"
        args = QCArgs.validate_namespace(qc_namespace)
        assert args.task == "rest"

    def test_validate_with_start_tr(self, qc_namespace: argparse.Namespace) -> None:
        """Test QCArgs preserves start_tr value."""
        qc_namespace.start_tr = 5
        args = QCArgs.validate_namespace(qc_namespace)
        assert args.start_tr == 5

    def test_validate_with_regressor(self, qc_namespace: argparse.Namespace) -> None:
        """Test QCArgs preserves regressor choice."""
        qc_namespace.regressor = "aCompCor"
        args = QCArgs.validate_namespace(qc_namespace)
        assert args.regressor == "aCompCor"

    def test_defaults(self, qc_namespace: argparse.Namespace) -> None:
        """Test default values."""
        args = QCArgs.validate_namespace(qc_namespace)
        assert args.task is None
        assert args.start_tr == 2
        assert args.regressor == "36-parameter"
        assert args.participant_label == []
        assert args.session_label == []


class TestQCRegistration:
    """Test that QC command is registered and discoverable."""

    def test_qc_command_registered(self) -> None:
        """Test QC workflow is available in parser."""
        result = cli(["/input", "/output", "qc", "--help"])
        assert result == 0

    def test_qc_parser_has_task(self) -> None:
        """Test QC subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc", "--task", "rest"])
        assert args.task == "rest"

    def test_qc_parser_has_start_tr(self) -> None:
        """Test QC subparser includes --start-tr argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc", "--start-tr", "5"])
        assert args.start_tr == 5

    def test_qc_parser_has_regressor(self) -> None:
        """Test QC subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc"])
        assert args.regressor == "36-parameter"

    def test_qc_parser_task_default_none(self) -> None:
        """Test QC subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc"])
        assert args.task is None

    def test_qc_parser_start_tr_default(self) -> None:
        """Test QC subparser --start-tr defaults to 2."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "qc"])
        assert args.start_tr == 2
