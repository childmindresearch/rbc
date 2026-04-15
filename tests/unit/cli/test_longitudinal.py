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
        args = FunctionalLongArgs.validate_namespace(base_ns)
        assert args.task == "rest"

    def test_invalid_task_rejected(self, base_ns: argparse.Namespace) -> None:
        """Task labels with special characters are rejected."""
        base_ns.task = "rest/invalid"
        with pytest.raises(ValueError, match="Task"):
            FunctionalLongArgs.validate_namespace(base_ns)


class TestMetricsLongArgs:
    """Tests for the metrics longitudinal subcommand validator."""

    def test_defaults(self, base_ns: argparse.Namespace) -> None:
        """FWHM defaults to 6 mm and atlas resolves from the registry."""
        base_ns.atlas = ["schaefer_200"]
        base_ns.fwhm = 6.0
        base_ns.task = None
        args = MetricsLongArgs.validate_namespace(base_ns)
        assert args.fwhm == 6.0
        assert "schaefer_200" in args.atlas_files

    def test_nonpositive_fwhm_rejected(self, base_ns: argparse.Namespace) -> None:
        """FWHM must be strictly positive."""
        base_ns.atlas = ["schaefer_200"]
        base_ns.fwhm = 0.0
        base_ns.task = None
        with pytest.raises(ValueError, match="FWHM"):
            MetricsLongArgs.validate_namespace(base_ns)


class TestAllLongArgs:
    """Tests for the combined longitudinal subcommand validator."""

    def test_defaults(self, base_ns: argparse.Namespace) -> None:
        """Defaults resolve to atlas registry + bundled 1 mm template."""
        base_ns.anat_template = None
        base_ns.atlas = ["schaefer_200"]
        base_ns.fwhm = 6.0
        args = AllLongArgs.validate_namespace(base_ns)
        assert args.fwhm == 6.0
        assert "schaefer_200" in args.atlas_files
        assert args.registration_template.name.endswith(".nii.gz")


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

    def test_metrics_subcommand_registers(self, tmp_path: Path) -> None:
        """Metrics subcommand is registered; Stage 6 raises NotImplementedError."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(NotImplementedError, match="Stage 6"):
            cli(["longitudinal", "metrics", str(input_dir), "-o", str(output_dir)])

    def test_qc_subcommand_registers(self, tmp_path: Path) -> None:
        """QC subcommand is registered; Stage 6 raises NotImplementedError."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(NotImplementedError, match="Stage 6"):
            cli(["longitudinal", "qc", str(input_dir), "-o", str(output_dir)])

    def test_all_subcommand_registers(self, tmp_path: Path) -> None:
        """All subcommand is registered; Stage 6 raises NotImplementedError."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(NotImplementedError, match="Stage 6"):
            cli(["longitudinal", "all", str(input_dir), "-o", str(output_dir)])
