"""Unit tests for Metrics CLI module."""

import argparse
from pathlib import Path

import pytest

from rbc.cli.main import cli, create_parser
from rbc.cli.metrics import MetricsArgs


class TestMetricsArgs:
    """Test suite for MetricsArgs validation."""

    @pytest.fixture
    def metrics_namespace(self, tmp_path: Path) -> argparse.Namespace:
        """Fixture for metrics argument namespace."""
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
            atlas="schaefer_200",
            fwhm=6.0,
            task=None,
            regressor="36-parameter",
        )

    def test_validate_namespace(self, metrics_namespace: argparse.Namespace) -> None:
        """Test MetricsArgs validates successfully with valid args."""
        args = MetricsArgs.validate_namespace(metrics_namespace)
        assert isinstance(args, MetricsArgs)
        assert args.atlas == "schaefer_200"
        assert args.fwhm == 6.0
        assert args.task is None
        assert args.regressor == "36-parameter"

    def test_validate_with_atlas(self, metrics_namespace: argparse.Namespace) -> None:
        """Test MetricsArgs preserves atlas choice."""
        metrics_namespace.atlas = "aal"
        args = MetricsArgs.validate_namespace(metrics_namespace)
        assert args.atlas == "aal"

    def test_validate_with_fwhm(self, metrics_namespace: argparse.Namespace) -> None:
        """Test MetricsArgs preserves fwhm value."""
        metrics_namespace.fwhm = 8.0
        args = MetricsArgs.validate_namespace(metrics_namespace)
        assert args.fwhm == 8.0

    def test_validate_with_task(self, metrics_namespace: argparse.Namespace) -> None:
        """Test MetricsArgs preserves task filter."""
        metrics_namespace.task = "rest"
        args = MetricsArgs.validate_namespace(metrics_namespace)
        assert args.task == "rest"

    def test_validate_with_regressor(
        self, metrics_namespace: argparse.Namespace
    ) -> None:
        """Test MetricsArgs preserves regressor choice."""
        metrics_namespace.regressor = "aCompCor"
        args = MetricsArgs.validate_namespace(metrics_namespace)
        assert args.regressor == "aCompCor"

    def test_defaults(self, metrics_namespace: argparse.Namespace) -> None:
        """Test default values."""
        args = MetricsArgs.validate_namespace(metrics_namespace)
        assert args.atlas == "schaefer_200"
        assert args.fwhm == 6.0
        assert args.task is None
        assert args.regressor == "36-parameter"
        assert args.participant_label == []
        assert args.session_label == []


class TestMetricsRegistration:
    """Test that metrics command is registered and discoverable."""

    def test_metrics_command_registered(self) -> None:
        """Test metrics workflow is available in parser."""
        result = cli(["/input", "/output", "metrics", "--help"])
        assert result == 0

    def test_metrics_parser_has_atlas(self) -> None:
        """Test metrics subparser includes --atlas argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.atlas == "schaefer_200"

    def test_metrics_parser_atlas_choices(self) -> None:
        """Test metrics subparser accepts valid atlas choices."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics", "--atlas", "aal"])
        assert args.atlas == "aal"

    def test_metrics_parser_has_fwhm(self) -> None:
        """Test metrics subparser includes --fwhm argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics", "--fwhm", "8.0"])
        assert args.fwhm == 8.0

    def test_metrics_parser_has_task(self) -> None:
        """Test metrics subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics", "--task", "rest"])
        assert args.task == "rest"

    def test_metrics_parser_has_regressor(self) -> None:
        """Test metrics subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.regressor == "36-parameter"

    def test_metrics_parser_task_default_none(self) -> None:
        """Test metrics subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.task is None

    def test_metrics_parser_fwhm_default(self) -> None:
        """Test metrics subparser --fwhm defaults to 6.0."""
        parser = create_parser()
        args = parser.parse_args(["/input", "/output", "metrics"])
        assert args.fwhm == 6.0
