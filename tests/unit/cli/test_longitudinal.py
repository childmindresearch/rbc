"""Unit tests for ``rbc longitudinal`` subcommand argparse wiring."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from rbc.cli.longitudinal._base import LongitudinalBaseArgs
from rbc.cli.longitudinal.all import AllLongArgs
from rbc.cli.longitudinal.anatomical import AnatomicalLongArgs
from rbc.cli.longitudinal.functional import FunctionalLongArgs
from rbc.cli.longitudinal.metrics import MetricsLongArgs
from rbc.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def base_ns(tmp_path: Path) -> argparse.Namespace:
    """Minimal argparse namespace shared across longitudinal subcommands."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    return argparse.Namespace(
        runner="local",
        verbose=0,
        input_dirs=[input_dir],
        output_dir=tmp_path / "output",
        participant_label=[],
        session_label=[],
        tmp_dir=None,
        ants_threads=1,
        fs_license=None,
    )


class TestLongitudinalBaseArgs:
    """Tests for the shared longitudinal base-args validator."""

    def test_no_license_is_allowed(self, base_ns: argparse.Namespace) -> None:
        """``--fs-license`` is optional; ``None`` is a valid default."""
        args = LongitudinalBaseArgs.validate_namespace(base_ns)
        assert args.fs_license is None

    def test_existing_license_accepted(
        self, base_ns: argparse.Namespace, tmp_path: Path
    ) -> None:
        """An existing ``--fs-license`` path round-trips through validation."""
        lic = tmp_path / "license.txt"
        lic.touch()
        base_ns.fs_license = lic
        args = LongitudinalBaseArgs.validate_namespace(base_ns)
        assert args.fs_license == lic

    def test_missing_license_rejected(
        self, base_ns: argparse.Namespace, tmp_path: Path
    ) -> None:
        """A non-existent ``--fs-license`` path raises ``ValueError``."""
        base_ns.fs_license = tmp_path / "nope.txt"
        with pytest.raises(ValueError, match="not found"):
            LongitudinalBaseArgs.validate_namespace(base_ns)


class TestAnatomicalLongArgs:
    """Tests for the anatomical longitudinal subcommand validator."""

    def test_defaults(self, base_ns: argparse.Namespace) -> None:
        """Anat template defaults to the bundled 1 mm registration template."""
        base_ns.anat_template = None
        args = AnatomicalLongArgs.validate_namespace(base_ns)
        assert args.registration_template.name.endswith(".nii.gz")


class TestFunctionalLongArgs:
    """Tests for the functional longitudinal subcommand validator."""

    def test_valid_task(self, base_ns: argparse.Namespace) -> None:
        """Alphanumeric task labels pass validation."""
        base_ns.task = "rest"
        base_ns.regressor = ["36-parameter"]
        args = FunctionalLongArgs.validate_namespace(base_ns)
        assert args.task == "rest"

    def test_invalid_task_rejected(self, base_ns: argparse.Namespace) -> None:
        """Task labels with special characters are rejected."""
        base_ns.task = "rest/invalid"
        base_ns.regressor = ["36-parameter"]
        with pytest.raises(ValueError, match="Task"):
            FunctionalLongArgs.validate_namespace(base_ns)

    def test_regressor_preserved(self, base_ns: argparse.Namespace) -> None:
        """Regressor choices round-trip through validation."""
        base_ns.task = None
        base_ns.regressor = ["36-parameter", "aCompCor"]
        args = FunctionalLongArgs.validate_namespace(base_ns)
        assert list(args.regressor) == ["36-parameter", "aCompCor"]


class TestMetricsLongArgs:
    """Tests for the metrics longitudinal subcommand validator."""

    def test_defaults(self, base_ns: argparse.Namespace) -> None:
        """Smooth defaults to None and atlas resolves from the registry."""
        base_ns.atlas = ["schaefer_200"]
        base_ns.smooth = None
        base_ns.tr = None
        base_ns.task = None
        base_ns.regressor = ["36-parameter"]
        args = MetricsLongArgs.validate_namespace(base_ns)
        assert args.smooth is None
        assert args.tr is None
        assert "schaefer_200" in args.atlas_files
        assert args.regressor == ["36-parameter"]

    def test_nonpositive_fwhm_rejected(self, base_ns: argparse.Namespace) -> None:
        """Smooth must be strictly positive."""
        base_ns.atlas = ["schaefer_200"]
        base_ns.smooth = 0.0
        base_ns.tr = None
        base_ns.task = None
        base_ns.regressor = ["36-parameter"]
        with pytest.raises(ValueError, match="smooth"):
            MetricsLongArgs.validate_namespace(base_ns)


class TestAllLongArgs:
    """Tests for the combined longitudinal subcommand validator."""

    def test_defaults(self, base_ns: argparse.Namespace) -> None:
        """Defaults resolve to atlas registry + bundled 1 mm template."""
        base_ns.anat_template = None
        base_ns.atlas = ["schaefer_200"]
        base_ns.smooth = None
        base_ns.regressor = ["36-parameter"]
        base_ns.task = None
        args = AllLongArgs.validate_namespace(base_ns)
        assert args.smooth is None
        assert "schaefer_200" in args.atlas_files
        assert args.registration_template.name.endswith(".nii.gz")
        assert args.regressor == ["36-parameter"]
        assert args.task is None


class TestParentSubparser:
    """Tests for ``rbc longitudinal --help`` and alias wiring."""

    def test_help_lists_all_stages(self, capsys: pytest.CaptureFixture[str]) -> None:
        """All six stages show up in the parent-subparser help output."""
        cli(["longitudinal", "--help"])
        out = capsys.readouterr().out
        for stage in ("template", "anatomical", "functional", "metrics", "qc", "all"):
            assert stage in out

    def test_long_alias_help_matches(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``rbc long --help`` exposes the same stages as ``rbc longitudinal``."""
        cli(["long", "--help"])
        out = capsys.readouterr().out
        for stage in ("template", "anatomical", "functional", "metrics", "qc", "all"):
            assert stage in out


class TestLongitudinalDispatch:
    """Tests for end-to-end argparse → orchestration wiring per subcommand."""

    def test_anatomical_dispatches(self, tmp_path: Path) -> None:
        """``rbc longitudinal anatomical`` routes to the anat orchestration."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.anatomical.run") as mock_run:
            rc = cli(
                [
                    "longitudinal",
                    "anatomical",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                ]
            )
            assert rc == 0
            mock_run.assert_called_once()
            kwargs = mock_run.call_args.kwargs
            assert kwargs["input_dirs"] == (input_dir,)
            assert kwargs["output_dir"] == output_dir

    def test_functional_dispatches(self, tmp_path: Path) -> None:
        """``rbc longitudinal functional`` routes to the func orchestration."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.functional.run") as mock_run:
            rc = cli(
                [
                    "longitudinal",
                    "functional",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                ]
            )
            assert rc == 0
            mock_run.assert_called_once()

    def test_metrics_dispatches(self, tmp_path: Path) -> None:
        """``rbc longitudinal metrics`` routes to the metrics orchestration."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.metrics.run") as mock_run:
            rc = cli(
                [
                    "longitudinal",
                    "metrics",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                ]
            )
            assert rc == 0
            mock_run.assert_called_once()

    def test_qc_dispatches(self, tmp_path: Path) -> None:
        """``rbc longitudinal qc`` routes to the QC orchestration."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.qc.run") as mock_run:
            rc = cli(
                [
                    "longitudinal",
                    "qc",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                ]
            )
            assert rc == 0
            mock_run.assert_called_once()

    def test_all_dispatches(self, tmp_path: Path) -> None:
        """``rbc longitudinal all`` routes to the all orchestration."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.all.run") as mock_run:
            rc = cli(
                [
                    "longitudinal",
                    "all",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                ]
            )
            assert rc == 0
            mock_run.assert_called_once()
