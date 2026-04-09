"""Unit tests for Metrics CLI module."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rbc.cli.main import cli, create_parser
from rbc.cli.metrics import MetricsArgs, _resolve_atlas_args


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base metrics argument namespace with sensible defaults."""
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
        atlas=["schaefer_200"],
        fwhm=6.0,
        regressor=["36-parameter"],
        tr=None,
        tmp_dir=None,
        ants_threads=1,
    )


class TestMetricsArgs:
    """Tests for MetricsArgs validation.

    Covers default values, custom values, preservation of all fields
    through validate_namespace, and input validation for task, fwhm,
    and atlas.
    """

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates a well-formed namespace."""
        args = MetricsArgs.validate_namespace(base_args)
        assert isinstance(args, MetricsArgs)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Default values for all fields are preserved through validation."""
        args = MetricsArgs.validate_namespace(base_args)
        assert args.task is None
        assert "schaefer_200" in args.atlas_files
        assert args.fwhm == 6.0
        assert args.regressor == ["36-parameter"]
        assert args.participant_label == []
        assert args.session_label == []

    @pytest.mark.parametrize("regressor", ["36-parameter", "aCompCor"])
    def test_valid_regressors(
        self, base_args: argparse.Namespace, regressor: str
    ) -> None:
        """Both supported regressor options pass validation."""
        base_args.regressor = [regressor]
        args = MetricsArgs.validate_namespace(base_args)
        assert args.regressor == [regressor]

    @pytest.mark.parametrize(
        "atlas",
        ["schaefer_200", "schaefer_300", "schaefer_400", "schaefer_1000", "aal"],
    )
    def test_valid_atlases(self, base_args: argparse.Namespace, atlas: str) -> None:
        """All supported atlas options pass validation."""
        base_args.atlas = [atlas]
        args = MetricsArgs.validate_namespace(base_args)
        assert atlas in args.atlas_files

    def test_invalid_atlas_raises(self, base_args: argparse.Namespace) -> None:
        """Unresolvable atlas name raises FileNotFoundError."""
        base_args.atlas = ["invalid_atlas"]
        with pytest.raises(FileNotFoundError):
            MetricsArgs.validate_namespace(base_args)

    @pytest.mark.parametrize("fwhm", [0.1, 1.0, 6.0, 10.0])
    def test_valid_fwhm(self, base_args: argparse.Namespace, fwhm: float) -> None:
        """Positive FWHM values pass validation."""
        base_args.fwhm = fwhm
        args = MetricsArgs.validate_namespace(base_args)
        assert args.fwhm == fwhm

    @pytest.mark.parametrize("fwhm", [0.0, -1.0, -6.0])
    def test_invalid_fwhm_raises(
        self, base_args: argparse.Namespace, fwhm: float
    ) -> None:
        """Zero or negative FWHM raises ValueError."""
        base_args.fwhm = fwhm
        with pytest.raises(ValueError, match="FWHM"):
            MetricsArgs.validate_namespace(base_args)

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
        args = MetricsArgs.validate_namespace(base_args)
        assert args.task == task

    @pytest.mark.parametrize(
        "task",
        ["faces n-back", "task label", "task!", "task/name"],
        ids=["space_hyphen", "space", "special_char", "slash"],
    )
    def test_invalid_task_labels_raise(
        self, base_args: argparse.Namespace, task: str
    ) -> None:
        """Invalid task labels raise ValueError."""
        base_args.task = task
        with pytest.raises(ValueError, match="Task must contain only alphanumeric"):
            MetricsArgs.validate_namespace(base_args)


class TestMetricsRegistration:
    """Test that metrics command is registered and discoverable."""

    def test_metrics_command_registered(self) -> None:
        """Test metrics workflow is available in parser."""
        result = cli(["metrics", "/input", "-o", "/output", "--help"])
        assert result == 0

    def test_metrics_parser_has_atlas(self) -> None:
        """Test metrics subparser includes --atlas argument."""
        parser = create_parser()
        args = parser.parse_args(["metrics", "/input", "-o", "/output"])
        assert args.atlas == ["schaefer_200"]

    def test_metrics_parser_atlas_choices(self) -> None:
        """Test metrics subparser accepts valid atlas choices."""
        parser = create_parser()
        args = parser.parse_args(
            ["metrics", "/input", "-o", "/output", "--atlas", "aal"]
        )
        assert args.atlas == ["aal"]

    def test_metrics_parser_has_fwhm(self) -> None:
        """Test metrics subparser includes --fwhm argument."""
        parser = create_parser()
        args = parser.parse_args(
            ["metrics", "/input", "-o", "/output", "--fwhm", "8.0"]
        )
        assert args.fwhm == 8.0

    def test_metrics_parser_has_task(self) -> None:
        """Test metrics subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(
            ["metrics", "/input", "-o", "/output", "--task", "rest"]
        )
        assert args.task == "rest"

    def test_metrics_parser_has_regressor(self) -> None:
        """Test metrics subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["metrics", "/input", "-o", "/output"])
        assert args.regressor == ["36-parameter"]

    def test_metrics_parser_task_default_none(self) -> None:
        """Test metrics subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["metrics", "/input", "-o", "/output"])
        assert args.task is None

    def test_metrics_parser_fwhm_default(self) -> None:
        """Test metrics subparser --fwhm defaults to 6.0."""
        parser = create_parser()
        args = parser.parse_args(["metrics", "/input", "-o", "/output"])
        assert args.fwhm == 6.0


class TestResolveAtlasArgs:
    """Tests for _resolve_atlas_args helper."""

    def test_registry_atlas(self) -> None:
        """Registry atlas names resolve to a label-to-path dict."""
        result = _resolve_atlas_args(["schaefer_200"])
        assert "schaefer_200" in result
        assert result["schaefer_200"].exists()

    def test_multiple_registry_atlases(self) -> None:
        """Multiple registry names are all resolved."""
        result = _resolve_atlas_args(["schaefer_200", "aal"])
        assert "schaefer_200" in result
        assert "aal" in result

    def test_custom_atlas_path(self, tmp_path: Path) -> None:
        """Custom NIfTI path resolves with stem as label."""
        atlas_file = tmp_path / "my_custom_atlas.nii.gz"
        atlas_file.touch()
        result = _resolve_atlas_args([str(atlas_file)])
        assert "my_custom_atlas" in result
        assert result["my_custom_atlas"] == atlas_file.resolve()

    def test_mixed_registry_and_custom(self, tmp_path: Path) -> None:
        """Registry names and custom paths can be mixed."""
        atlas_file = tmp_path / "custom.nii.gz"
        atlas_file.touch()
        result = _resolve_atlas_args(["schaefer_200", str(atlas_file)])
        assert "schaefer_200" in result
        assert "custom" in result

    def test_nonexistent_custom_path_raises(self) -> None:
        """Non-existent custom atlas path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _resolve_atlas_args(["/nonexistent/atlas.nii.gz"])

    def test_duplicate_label_raises(self, tmp_path: Path) -> None:
        """Duplicate labels from different custom paths raise ValueError."""
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        (d1 / "dup.nii.gz").touch()
        (d2 / "dup.nii.gz").touch()
        with pytest.raises(ValueError, match="Duplicate atlas label"):
            _resolve_atlas_args([str(d1 / "dup.nii.gz"), str(d2 / "dup.nii.gz")])
