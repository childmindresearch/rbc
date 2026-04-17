"""Unit tests for ``rbc.orchestration.longitudinal.template``."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from rbc.orchestration import Filters
from rbc.orchestration.longitudinal.template import setup_freesurfer_auth, run

if TYPE_CHECKING:
    from pathlib import Path


_SCHEMA = [
    "datatype",
    "suffix",
    "ext",
    "sub",
    "ses",
    "space",
    "task",
    "run",
    "desc",
    "root",
    "path",
]


def _brain_row(sub: str, ses: str) -> tuple:
    path = f"sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_desc-brain_T1w.nii.gz"
    return (
        "anat",
        "T1w",
        ".nii.gz",
        sub,
        ses,
        None,
        None,
        None,
        "brain",
        "/data",
        path,
    )


def _df(*rows: tuple) -> pl.DataFrame:
    return pl.DataFrame(dict(zip(_SCHEMA, zip(*rows, strict=True), strict=True)))


class TestSetupFreesurferAuth:
    """Tests for the FS license / SURFER_SIDEDOOR fallback chain."""

    def test_explicit_license_mounted(self, tmp_path: Path) -> None:
        """An explicit license path takes precedence and is mounted."""
        license_path = tmp_path / "license.txt"
        license_path.touch()
        runner = MagicMock()

        with (
            patch(
                "rbc.orchestration.longitudinal.template.niwrap.get_global_runner",
                return_value=runner,
            ),
            patch(
                "rbc.orchestration.longitudinal.template.mount_fs_license"
            ) as mock_mount,
        ):
            setup_freesurfer_auth(license_path)
            mock_mount.assert_called_once_with(runner, license_path)

    def test_env_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FS_LICENSE env var is used when --fs-license is absent."""
        license_path = tmp_path / "env_license.txt"
        license_path.touch()
        monkeypatch.setenv("FS_LICENSE", str(license_path))
        runner = MagicMock()

        with (
            patch(
                "rbc.orchestration.longitudinal.template.niwrap.get_global_runner",
                return_value=runner,
            ),
            patch(
                "rbc.orchestration.longitudinal.template.mount_fs_license"
            ) as mock_mount,
        ):
            setup_freesurfer_auth(None)
            mock_mount.assert_called_once_with(runner, license_path)

    def test_bypass_when_no_license(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No license + no env var sets SURFER_SIDEDOOR on the runner."""
        monkeypatch.delenv("FS_LICENSE", raising=False)
        runner = MagicMock(spec=["environ"])
        runner.environ = {}

        with (
            patch(
                "rbc.orchestration.longitudinal.template.niwrap.get_global_runner",
                return_value=runner,
            ),
            patch(
                "rbc.orchestration.longitudinal.template.mount_fs_license"
            ) as mock_mount,
        ):
            setup_freesurfer_auth(None)
            assert runner.environ["SURFER_SIDEDOOR"] == "1"
            mock_mount.assert_not_called()

    def test_bypass_skipped_when_runner_lacks_environ(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Runners without an ``environ`` attr are tolerated."""
        monkeypatch.delenv("FS_LICENSE", raising=False)
        runner = object()  # no environ attr

        with (
            patch(
                "rbc.orchestration.longitudinal.template.niwrap.get_global_runner",
                return_value=runner,
            ),
            patch(
                "rbc.orchestration.longitudinal.template.mount_fs_license"
            ) as mock_mount,
        ):
            setup_freesurfer_auth(None)
            mock_mount.assert_not_called()

    def test_missing_license_raises(self, tmp_path: Path) -> None:
        """A non-existent license path raises before reaching the runner."""
        with pytest.raises(FileNotFoundError, match="not found"):
            setup_freesurfer_auth(tmp_path / "missing.txt")


class TestRunSkipWarning:
    """Tests for warn-and-skip behavior on single-session subjects."""

    def test_warns_on_skipped_subject(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Single-session subjects trigger a warning but don't abort."""
        df = _df(
            _brain_row("01", "baseline"),
            _brain_row("01", "vis2"),
            _brain_row("02", "baseline"),
        )
        with (
            patch("rbc.orchestration.longitudinal.template.init_runner"),
            patch("rbc.orchestration.longitudinal.template.setup_freesurfer_auth"),
            patch(
                "rbc.orchestration.longitudinal.template.load_table",
                return_value=df,
            ),
            patch(
                "rbc.orchestration.longitudinal.template.process_subject"
            ) as mock_process,
            caplog.at_level(logging.WARNING),
        ):
            run(input_dirs=[tmp_path], output_dir=tmp_path, filters=Filters())
            mock_process.assert_called_once()
            assert any("sub-02" in msg for msg in caplog.messages)
