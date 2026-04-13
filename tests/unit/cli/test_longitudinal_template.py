"""Unit tests for the ``rbc longitudinal template`` CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

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
                [
                    "longitudinal",
                    "template",
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

    def test_long_alias_dispatches(self, tmp_path: Path) -> None:
        """The ``long`` alias resolves to the same subparser tree."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with patch("rbc.cli.longitudinal.template.run") as mock_run:
            rc = cli(
                ["long", "template", str(input_dir), "-o", str(output_dir)]
            )
            assert rc == 0
            mock_run.assert_called_once()
