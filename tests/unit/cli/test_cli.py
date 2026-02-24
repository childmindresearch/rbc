"""Unit tests for CLI module."""

import argparse
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from rbc.cli import main as cli


class TestGlobalOpts:
    """Testing suite for global options in parser."""

    @pytest.mark.parametrize("subject", ["sub-01", "01"])
    def test_participant_labels(self, subject: str) -> None:
        """Tests participant label correctly grabbed and strips prefix if needed."""
        parser = cli._global_opts()
        args = parser.parse_args(["--participant-label", subject])
        assert args.participant_label == ["01"]

    @pytest.mark.parametrize("session", ["ses-baseline", "baseline"])
    def test_session_label(self, session: str) -> None:
        """Tests session label correctly grabbed and strips prefix if needed."""
        parser = cli._global_opts()
        args = parser.parse_args(["--session-label", session])
        assert args.session_label == ["baseline"]

    def test_default_runner(self) -> None:
        """Tests runner default to 'local'."""
        parser = cli._global_opts()
        args = parser.parse_args([])
        assert args.runner == "local"
        assert args.verbose == 0

    @pytest.mark.parametrize(
        "runner", ["local", "LOCAL", "docker", "dOcKeR", "Singularity", "singularity"]
    )
    def test_valid_runner(self, runner: str) -> None:
        """Tests runner argument accepts valid choices."""
        parser = cli._global_opts()
        args = parser.parse_args(["--runner", runner])
        assert args.runner == runner.lower()

    def test_invalid_runner(self) -> None:
        """Test runner rejects invalid choice."""
        parser = cli._global_opts()
        with pytest.raises(SystemExit):
            parser.parse_args(["--runner", "invalid"])

    @pytest.mark.parametrize(
        ("verbosity", "log_count"), [("-v", 1), ("-vv", 2), ("-vvv", 3)]
    )
    def test_verbosity(self, verbosity: str, log_count: int) -> None:
        """Test verbosity option."""
        parser = cli._global_opts()
        args = parser.parse_args([verbosity])
        assert args.verbose == log_count


class TestParser:
    """Testing suite for parser."""

    def test_parser_inherits_global_opts(self) -> None:
        """Test parser inherits global options."""
        parser = cli.create_parser()
        dest_names = {action.dest for action in parser._actions}
        assert "participant_label" in dest_names
        assert "session_label" in dest_names
        assert "runner" in dest_names


class TestCLI:
    """Testing suite for main CLI function."""

    def test_error_handling(self) -> None:
        """Test CLI returns error code on parsing failure."""
        result = cli.cli(["--invalid"])
        assert isinstance(result, int)
        assert result != 0

    @patch("rbc.cli.anatomical.register_command")
    def test_calls_func(self, mock_register: Mock) -> None:
        """Test that CLI calls subparser function if it exists."""
        mock_func = Mock(return_value=0)

        def register_side_effect(
            subparsers: argparse._SubParsersAction,
            **kwargs: Any,  # noqa: ARG001, ANN401
        ) -> None:
            parser = subparsers.add_parser("anatomical")
            parser.set_defaults(func=mock_func)

        mock_register.side_effect = register_side_effect
        result = cli.cli(["input", "output", "anatomical"])
        assert mock_func.called
        assert result == 0

    @patch("rbc.cli.anatomical.register_command")
    def test_global_opts_propagate_to_workflow(self, mock_register: Mock) -> None:
        """Test that global options available to workflow function."""
        received_args = None

        def mock_func(args: Any, **kwargs: Any) -> int:  # noqa: ARG001, ANN401
            nonlocal received_args
            received_args = args
            return 0

        def register_side_effect(
            subparsers: argparse._SubParsersAction,
            **kwargs: Any,  # noqa: ANN401
        ) -> None:
            parents = kwargs.get("parents", [])
            parser = subparsers.add_parser("anatomical", parents=parents)
            parser.set_defaults(func=mock_func)

        mock_register.side_effect = register_side_effect

        cli.cli(["input", "output", "anatomical", "--participant-label", "01"])
        assert received_args is not None
        assert received_args.participant_label == ["01"]

    @patch("rbc.cli.anatomical.register_command")
    def test_cli_prints_help_without_func(
        self, mock_register: Mock, capsys: pytest.CaptureFixture
    ) -> None:
        """CLI should print help and return 1 if workflow has no func attribute."""

        # Register command without setting func
        def register_side_effect(
            subparsers: argparse._SubParsersAction,
            **kwargs: Any,  # noqa: ARG001, ANN401
        ) -> None:
            subparsers.add_parser("anatomical")

        mock_register.side_effect = register_side_effect

        result = cli.cli(["/input", "/output", "anatomical"])
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
        assert result == 1


class TestValidation:
    """Test suite for validating namespace into NamedTuple."""

    @pytest.fixture
    def base_args(self, tmp_path: Path) -> argparse.Namespace:
        """Fixture for base argument namespace."""
        input_dir = tmp_path / "input"
        input_dir.touch()
        output_dir = tmp_path / "output"
        return argparse.Namespace(
            runner="local",
            verbose=False,
            input_dir=input_dir,
            output_dir=output_dir,
            participant_label=[],
            session_label=[],
        )

    def test_valid(self, base_args: argparse.Namespace) -> None:
        """Test validation succeeds."""
        args = cli.BaseArgs.validate_namespace(base_args)
        assert isinstance(args, cli.BaseArgs)

    def test_no_input_dir(self, base_args: argparse.Namespace) -> None:
        """Test error raised if input path doesn't exist."""
        base_args.input_dir = Path("invalid")
        with pytest.raises(ValueError, match="Input path does not exist"):
            cli.BaseArgs.validate_namespace(base_args)

    def test_invalid_runner(self, base_args: argparse.Namespace) -> None:
        """Test error raised if runner not valid."""
        base_args.runner = "invalid"
        with pytest.raises(ValueError, match="Expected one of"):
            cli.BaseArgs.validate_namespace(base_args)

    def test_invalid_participant_label(self, base_args: argparse.Namespace) -> None:
        """Test error raised if participant prefix is incorrect."""
        base_args.participant_label = ["sub-01"]
        with pytest.raises(ValueError, match="Label must not start with"):
            cli.BaseArgs.validate_namespace(base_args)

    def test_invalid_session_label(self, base_args: argparse.Namespace) -> None:
        """Test error raised if session prefix is incorrect."""
        base_args.session_label = ["ses-01"]
        with pytest.raises(ValueError, match="Label must not start with"):
            cli.BaseArgs.validate_namespace(base_args)


class TestMain:
    """Test main entry point."""

    @patch("rbc.cli.main.cli")
    @patch.object(sys, "exit")
    def test_main_successful(self, mock_exit: Mock, mock_cli: Mock) -> None:
        """Test cli successfully called."""
        mock_cli.return_value = 0
        cli.main()
        mock_cli.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("rbc.cli.main.cli")
    @patch.object(sys, "exit")
    def test_main_error(self, mock_exit: Mock, mock_cli: Mock) -> None:
        """Test cli unsuccessfully called."""
        mock_cli.return_value = 1
        cli.main()
        mock_exit.assert_called_once_with(1)
