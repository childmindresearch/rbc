"""Unit tests for All (combined pipeline) CLI module."""

import argparse
from pathlib import Path

import pytest

from rbc.cli.all import AllArgs
from rbc.cli.main import cli, create_parser


class TestAllArgs:
    """Test suite for AllArgs validation."""

    @pytest.fixture
    def all_namespace(self, tmp_path: Path) -> argparse.Namespace:
        """Fixture for all-pipeline argument namespace."""
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
            regressor="36-parameter",
            task=None,
            atlas="schaefer_200",
            fwhm=6.0,
            start_tr=2,
        )

    def test_validate_namespace(self, all_namespace: argparse.Namespace) -> None:
        """Test AllArgs validates successfully with valid args."""
        args = AllArgs.validate_namespace(all_namespace)
        assert isinstance(args, AllArgs)
        assert args.regressor == "36-parameter"
        assert args.task is None
        assert args.atlas == "schaefer_200"
        assert args.fwhm == 6.0
        assert args.start_tr == 2

    def test_validate_with_regressor(self, all_namespace: argparse.Namespace) -> None:
        """Test AllArgs preserves regressor choice."""
        all_namespace.regressor = "aCompCor"
        args = AllArgs.validate_namespace(all_namespace)
        assert args.regressor == "aCompCor"

    def test_validate_with_task(self, all_namespace: argparse.Namespace) -> None:
        """Test AllArgs preserves task filter."""
        all_namespace.task = "rest"
        args = AllArgs.validate_namespace(all_namespace)
        assert args.task == "rest"

    def test_validate_with_atlas(self, all_namespace: argparse.Namespace) -> None:
        """Test AllArgs preserves atlas choice."""
        all_namespace.atlas = "aal"
        args = AllArgs.validate_namespace(all_namespace)
        assert args.atlas == "aal"

    def test_validate_with_fwhm(self, all_namespace: argparse.Namespace) -> None:
        """Test AllArgs preserves fwhm value."""
        all_namespace.fwhm = 8.0
        args = AllArgs.validate_namespace(all_namespace)
        assert args.fwhm == 8.0

    def test_validate_with_start_tr(self, all_namespace: argparse.Namespace) -> None:
        """Test AllArgs preserves start_tr value."""
        all_namespace.start_tr = 5
        args = AllArgs.validate_namespace(all_namespace)
        assert args.start_tr == 5

    def test_defaults(self, all_namespace: argparse.Namespace) -> None:
        """Test default values."""
        args = AllArgs.validate_namespace(all_namespace)
        assert args.regressor == "36-parameter"
        assert args.task is None
        assert args.atlas == "schaefer_200"
        assert args.fwhm == 6.0
        assert args.start_tr == 2
        assert args.participant_label == []
        assert args.session_label == []


class TestAllRegistration:
    """Test that all command is registered and discoverable."""

    def test_all_command_registered(self) -> None:
        """Test all workflow is available in parser."""
        result = cli(["/input", "/output", "all", "--help"])
        assert result == 0

    def test_all_parser_has_regressor(self) -> None:
        """Test all subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.regressor == "36-parameter"

    def test_all_parser_has_task(self) -> None:
        """Test all subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--task", "rest"])
        assert args.task == "rest"

    def test_all_parser_has_atlas(self) -> None:
        """Test all subparser includes --atlas argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.atlas == "schaefer_200"

    def test_all_parser_atlas_choices(self) -> None:
        """Test all subparser accepts valid atlas choices."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--atlas", "aal"])
        assert args.atlas == "aal"

    def test_all_parser_has_fwhm(self) -> None:
        """Test all subparser includes --fwhm argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--fwhm", "8.0"])
        assert args.fwhm == 8.0

    def test_all_parser_has_start_tr(self) -> None:
        """Test all subparser includes --start-tr argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all", "--start-tr", "5"])
        assert args.start_tr == 5

    def test_all_parser_task_default_none(self) -> None:
        """Test all subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.task is None

    def test_all_parser_fwhm_default(self) -> None:
        """Test all subparser --fwhm defaults to 6.0."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.fwhm == 6.0

    def test_all_parser_start_tr_default(self) -> None:
        """Test all subparser --start-tr defaults to 2."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "all"])
        assert args.start_tr == 2
