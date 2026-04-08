"""Unit tests for All (combined pipeline) CLI module."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rbc.cli.all import AllArgs
from rbc.cli.main import cli, create_parser


@pytest.fixture
def base_args(tmp_path: Path) -> argparse.Namespace:
    """Fixture for base all-pipeline argument namespace with sensible defaults."""
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
        regressor=["36-parameter"],
        task=None,
        atlas=["schaefer_200"],
        fwhm=6.0,
        start_tr=2,
        tr=None,
        tmp_dir=None,
        brain_extraction_template=None,
        brain_extraction_prob_mask=None,
        brain_extraction_reg_mask=None,
        anat_template=None,
        func_template=None,
        func_template_mask=None,
        func_template_ref=None,
    )


class TestAllArgs:
    """Tests for AllArgs validation.

    Covers default values, custom values, preservation of all fields
    through validate_namespace, and input validation for task, fwhm,
    atlas, and start_tr.
    """

    def test_parser_from_namespace(self, base_args: argparse.Namespace) -> None:
        """Tests parser successfully validates a well-formed namespace."""
        args = AllArgs.validate_namespace(base_args)
        assert isinstance(args, AllArgs)

    def test_defaults(self, base_args: argparse.Namespace) -> None:
        """Default values for all fields are preserved through validation."""
        args = AllArgs.validate_namespace(base_args)
        assert args.regressor == ["36-parameter"]
        assert args.task is None
        assert "schaefer_200" in args.atlas_files
        assert args.fwhm == 6.0
        assert args.start_tr == 2
        assert args.participant_label == []
        assert args.session_label == []

    @pytest.mark.parametrize("regressor", ["36-parameter", "aCompCor"])
    def test_valid_regressors(
        self, base_args: argparse.Namespace, regressor: str
    ) -> None:
        """Both supported regressor options pass validation."""
        base_args.regressor = [regressor]
        args = AllArgs.validate_namespace(base_args)
        assert args.regressor == [regressor]

    @pytest.mark.parametrize(
        "atlas",
        ["schaefer_200", "schaefer_300", "schaefer_400", "schaefer_1000", "aal"],
    )
    def test_valid_atlases(self, base_args: argparse.Namespace, atlas: str) -> None:
        """All supported atlas options pass validation."""
        base_args.atlas = [atlas]
        args = AllArgs.validate_namespace(base_args)
        assert atlas in args.atlas_files

    def test_invalid_atlas_raises(self, base_args: argparse.Namespace) -> None:
        """Unresolvable atlas name raises FileNotFoundError."""
        base_args.atlas = ["invalid_atlas"]
        with pytest.raises(FileNotFoundError):
            AllArgs.validate_namespace(base_args)

    @pytest.mark.parametrize("fwhm", [0.1, 1.0, 6.0, 10.0])
    def test_valid_fwhm(self, base_args: argparse.Namespace, fwhm: float) -> None:
        """Positive FWHM values pass validation."""
        base_args.fwhm = fwhm
        args = AllArgs.validate_namespace(base_args)
        assert args.fwhm == fwhm

    @pytest.mark.parametrize("fwhm", [0.0, -1.0, -6.0])
    def test_invalid_fwhm_raises(
        self, base_args: argparse.Namespace, fwhm: float
    ) -> None:
        """Zero or negative FWHM raises ValueError."""
        base_args.fwhm = fwhm
        with pytest.raises(ValueError, match="FWHM"):
            AllArgs.validate_namespace(base_args)

    def test_custom_start_tr(self, base_args: argparse.Namespace) -> None:
        """Custom start_tr is correctly preserved through validation."""
        base_args.start_tr = 5
        args = AllArgs.validate_namespace(base_args)
        assert args.start_tr == 5

    @pytest.mark.parametrize("start_tr", [0, -1, -5])
    def test_invalid_start_tr_raises(
        self, base_args: argparse.Namespace, start_tr: int
    ) -> None:
        """Zero or negative start_tr raises ValueError."""
        base_args.start_tr = start_tr
        with pytest.raises(ValueError, match="Start TR"):
            AllArgs.validate_namespace(base_args)

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
        args = AllArgs.validate_namespace(base_args)
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
            AllArgs.validate_namespace(base_args)


class TestAllRegistration:
    """Test that all command is registered and discoverable."""

    def test_all_command_registered(self) -> None:
        """Test all workflow is available in parser."""
        result = cli(["all", "/input", "-o", "/output", "--help"])
        assert result == 0

    def test_all_parser_has_regressor(self) -> None:
        """Test all subparser includes --regressor argument."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output"])
        assert args.regressor == ["36-parameter"]

    def test_all_parser_has_task(self) -> None:
        """Test all subparser includes --task argument."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output", "--task", "rest"])
        assert args.task == "rest"

    def test_all_parser_has_atlas(self) -> None:
        """Test all subparser includes --atlas argument."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output"])
        assert args.atlas == ["schaefer_200"]

    def test_all_parser_atlas_choices(self) -> None:
        """Test all subparser accepts valid atlas choices."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output", "--atlas", "aal"])
        assert args.atlas == ["aal"]

    def test_all_parser_has_fwhm(self) -> None:
        """Test all subparser includes --fwhm argument."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output", "--fwhm", "8.0"])
        assert args.fwhm == 8.0

    def test_all_parser_has_start_tr(self) -> None:
        """Test all subparser includes --start-tr argument."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output", "--start-tr", "5"])
        assert args.start_tr == 5

    def test_all_parser_task_default_none(self) -> None:
        """Test all subparser --task defaults to None."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output"])
        assert args.task is None

    def test_all_parser_fwhm_default(self) -> None:
        """Test all subparser --fwhm defaults to 6.0."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output"])
        assert args.fwhm == 6.0

    def test_all_parser_start_tr_default(self) -> None:
        """Test all subparser --start-tr defaults to 2."""
        parser = create_parser()
        args = parser.parse_args(["all", "/input", "-o", "/output"])
        assert args.start_tr == 2
