"""Unit tests for Niwrap helpers."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest
from niwrap import (
    DockerRunner,
    LocalRunner,
    Runner,
    SingularityRunner,
)
from styxpodman import PodmanRunner

from rbc.core.niwrap import generate_exec_folder, setup_runner

if TYPE_CHECKING:
    from pathlib import Path


class TestSetupRunner:
    """Test suite for niwrap.setup_runner."""

    @pytest.fixture(autouse=True)
    def _no_styxcache_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # setup_runner wraps with a CachingRunner when RBC_STYXCACHE_DIR is set.
        # Tests here probe the concrete runner type, so drop the env var.
        monkeypatch.delenv("RBC_STYXCACHE_DIR", raising=False)

    def test_default(self, tmp_path: Path) -> None:
        """Test default initialization uses auto-detection."""
        ctx = setup_runner(tmp_dir=tmp_path)
        assert isinstance(ctx.logger, logging.Logger)
        assert ctx.runner is not None

    @pytest.mark.parametrize(
        ("runner", "runner_type"),
        [
            ("local", LocalRunner),
            ("docker", DockerRunner),
            ("podman", PodmanRunner),
            pytest.param(
                "singularity",
                SingularityRunner,
                marks=pytest.mark.skipif(
                    os.name == "nt",
                    reason="SingularityRunner not supported on Windows",
                ),
            ),
        ],
    )
    def test_set_runner(
        self, runner: str, runner_type: type[Runner], tmp_path: Path
    ) -> None:
        """Test explicit setting of runner."""
        ctx = setup_runner(runner=runner, tmp_dir=tmp_path)  # type: ignore [arg-type]
        assert isinstance(ctx.runner, runner_type)

    def test_invalid_runner(self, tmp_path: Path) -> None:
        """Test error raised if invalid runner selected."""
        with pytest.raises(NotImplementedError, match="Unknown runner"):
            setup_runner(runner="invalid", tmp_dir=tmp_path)  # type: ignore [arg-type]

    def test_set_tmp_dir(self, tmp_path: Path) -> None:
        """Test setting of data directory works."""
        ctx = setup_runner(tmp_dir=tmp_path)
        assert ctx.runner.data_dir.is_relative_to(tmp_path)
        assert ctx.runner.data_dir.exists()

    @pytest.mark.parametrize(
        ("verbose", "log_level"),
        [
            (0, logging.WARNING),
            (1, logging.INFO),
            (2, logging.DEBUG),
            (5, logging.DEBUG),
        ],
    )
    def test_set_log_level(self, verbose: int, log_level: int, tmp_path: Path) -> None:
        """Test setting of log levels."""
        ctx = setup_runner(verbose=verbose, tmp_dir=tmp_path)
        assert ctx.logger.level == log_level


class TestGenExecFolder:
    """Testing suite for niwrap.generate_exec_folder."""

    def test_create_folder_default(self) -> None:
        """Test folder successfully generated with default arguments."""
        result = generate_exec_folder()
        assert result.exists()
        assert result.is_dir()
        assert result.name.startswith("python_")

    def test_create_folder_with_suffix(self) -> None:
        """Test folder successfully generates with suffix prefix."""
        result = generate_exec_folder(suffix="pytest")
        assert result.name.startswith("pytest_")

    def test_folders_are_unique(self) -> None:
        """Each call returns a distinct path, even with the same suffix."""
        a = generate_exec_folder("same")
        b = generate_exec_folder("same")
        assert a != b
