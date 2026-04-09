"""Unit tests for Functional CLI module."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from rbc.cli.functional import FunctionalArgs
from rbc_resources import REGISTRATION_TEMPLATES


class TestFunctionalArgs:
    """Tests for FunctionalArgs validation."""

    @pytest.fixture
    def func_namespace(self, tmp_path: Path) -> argparse.Namespace:
        """Fixture for functional argument namespace."""
        input_dir = tmp_path / "input"
        input_dir.touch()
        output_dir = tmp_path / "output"
        return argparse.Namespace(
            runner="local",
            verbose=0,
            input_dirs=[input_dir],
            output_dir=output_dir,
            participant_label=[],
            session_label=[],
            regressor=["36-parameter"],
            task=None,
            tr=None,
            tmp_dir=None,
            func_template=None,
            func_template_mask=None,
            func_template_ref=None,
            ants_threads=1,
        )

    def test_validate_namespace(self, func_namespace: argparse.Namespace) -> None:
        """Test FunctionalArgs validates successfully with valid args."""
        args = FunctionalArgs.validate_namespace(func_namespace)
        assert isinstance(args, FunctionalArgs)
        assert args.regressor == ["36-parameter"]
        assert args.task is None

    def test_validate_with_regressor(self, func_namespace: argparse.Namespace) -> None:
        """Test FunctionalArgs preserves regressor choice."""
        func_namespace.regressor = ["aCompCor"]
        args = FunctionalArgs.validate_namespace(func_namespace)
        assert args.regressor == ["aCompCor"]

    def test_validate_with_task(self, func_namespace: argparse.Namespace) -> None:
        """Test FunctionalArgs preserves task filter."""
        func_namespace.task = "rest"
        args = FunctionalArgs.validate_namespace(func_namespace)
        assert args.task == "rest"

    def test_defaults(self, func_namespace: argparse.Namespace) -> None:
        """Test default values for regressor and task."""
        args = FunctionalArgs.validate_namespace(func_namespace)
        assert args.regressor == ["36-parameter"]
        assert args.task is None
        assert args.participant_label == []
        assert args.session_label == []

    def test_parser_from_namespace(self, func_namespace: argparse.Namespace) -> None:
        """Tests parser successfully validates namespace."""
        args = FunctionalArgs.validate_namespace(func_namespace)
        assert isinstance(args, FunctionalArgs)

    @pytest.mark.parametrize(
        "task",
        ["rest", "nback", "faces+n+back", "task123", None],
        ids=["simple", "alphanumeric", "plus_separator", "with_digits", "none"],
    )
    def test_valid_task_labels(
        self, func_namespace: argparse.Namespace, task: str | None
    ) -> None:
        """Tests valid task labels pass validation."""
        func_namespace.task = task
        args = FunctionalArgs.validate_namespace(func_namespace)
        assert args.task == task

    @pytest.mark.parametrize(
        "task",
        ["faces n-back", "task label", "task!", "task/name"],
        ids=["space_hyphen", "space", "special_char", "slash"],
    )
    def test_invalid_task_labels(
        self, func_namespace: argparse.Namespace, task: str
    ) -> None:
        """Tests invalid task labels raise ValueError."""
        func_namespace.task = task
        with pytest.raises(ValueError, match="Task must contain only alphanumeric"):
            FunctionalArgs.validate_namespace(func_namespace)

    def test_defaults_use_bundled_templates(
        self, func_namespace: argparse.Namespace
    ) -> None:
        """When all template args are None, bundled defaults are used."""
        args = FunctionalArgs.validate_namespace(func_namespace)
        assert args.func_template == REGISTRATION_TEMPLATES.brain_2mm
        assert args.func_template_mask == REGISTRATION_TEMPLATES.brain_mask_2mm
        assert args.func_template_ref == REGISTRATION_TEMPLATES.bold_ref
