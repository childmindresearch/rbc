"""Unit tests for the ``rbc longitudinal template`` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from rbc.cli.main import cli

if TYPE_CHECKING:
    from pathlib import Path


class TestTemplateSubcommand:
    """Tests for the ``rbc longitudinal template`` argparse wiring."""

    def test_template_dispatches_to_run(self, tmp_path: Path) -> None:
        """Parsing routes to orchestration.longitudinal.template.run."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.template.run") as mock_run:
            rc = cli(
                ["longitudinal", "template", str(input_dir), "-o", str(output_dir)]
            )
            assert rc == 0
            mock_run.assert_called_once()
            kwargs = mock_run.call_args.kwargs
            assert kwargs["input_dirs"] == (input_dir,)
            assert kwargs["output_dir"] == output_dir
            assert kwargs["fs_license"] is None

    def test_long_alias_dispatches(self, tmp_path: Path) -> None:
        """The ``long`` alias resolves to the same subparser tree."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.template.run") as mock_run:
            rc = cli(["long", "template", str(input_dir), "-o", str(output_dir)])
            assert rc == 0
            mock_run.assert_called_once()

    def test_explicit_fs_license_passed_through(self, tmp_path: Path) -> None:
        """An explicit --fs-license is forwarded to run()."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        license_path = tmp_path / "license.txt"
        license_path.touch()

        with patch("rbc.cli.longitudinal.template.run") as mock_run:
            rc = cli(
                [
                    "longitudinal",
                    "template",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--fs-license",
                    str(license_path),
                ]
            )
            assert rc == 0
            assert mock_run.call_args.kwargs["fs_license"] == license_path

    def test_nonexistent_fs_license_rejected(self, tmp_path: Path) -> None:
        """A --fs-license path that doesn't exist raises before dispatch."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        missing = tmp_path / "missing.txt"

        with (
            patch("rbc.cli.longitudinal.template.run") as mock_run,
            pytest.raises(ValueError, match="not found"),
        ):
            cli(
                [
                    "longitudinal",
                    "template",
                    str(input_dir),
                    "-o",
                    str(output_dir),
                    "--fs-license",
                    str(missing),
                ]
            )
        mock_run.assert_not_called()
