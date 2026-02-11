"""Unit tests for Niwrap helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from niwrap import DockerRunner, LocalRunner, Runner, SingularityRunner

from rbc.core.niwrap import setup_runner

if TYPE_CHECKING:
    from pathlib import Path


class TestSetupRunner:
    """Test suite for niwrap.setup_runner."""

    def test_default(self, tmp_path: Path) -> None:
        """Test default initialization."""
        ctx = setup_runner()
        assert isinstance(ctx.logger, logging.Logger)
        assert isinstance(ctx.runner, LocalRunner)
        assert ctx.runner.data_dir != tmp_path

    @pytest.mark.parametrize(
        ("runner", "runner_type"),
        [
            ("local", LocalRunner),
            ("docker", DockerRunner),
            ("singularity", SingularityRunner),
        ],
    )
    def test_set_runner(self, runner: str, runner_type: type[Runner]) -> None:
        """Test explicit setting of runner."""
        ctx = setup_runner(runner=runner)
        assert isinstance(ctx.runner, runner_type)

    def test_invalid_runner(self) -> None:
        """Test error raised if invalid runner selected."""
        with pytest.raises(NotImplementedError, match="Unknown runner"):
            setup_runner(runner="invalid")

    def test_set_tmp_dir(self, tmp_path: Path) -> None:
        """Test setting of data directory works."""
        ctx = setup_runner(tmp_dir=tmp_path)
        assert ctx.runner.data_dir == tmp_path
