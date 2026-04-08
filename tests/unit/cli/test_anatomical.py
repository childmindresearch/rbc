"""Unit tests for Anatomical CLI module."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest

from rbc.cli import anatomical
from rbc_resources import BRAIN_EXTRACTION_TEMPLATES, REGISTRATION_TEMPLATES

if TYPE_CHECKING:
    from pathlib import Path


class TestAnatomicalArgs:
    """Tests for AnatomicalArgs validation."""

    @pytest.fixture
    def anat_namespace(self, tmp_path: Path) -> argparse.Namespace:
        """Fixture for anatomical argument namespace."""
        input_dir = tmp_path / "input"
        input_dir.touch()
        return argparse.Namespace(
            runner="local",
            verbose=False,
            input_dir=input_dir,
            output_dir=tmp_path / "output",
            participant_label=[],
            session_label=[],
            tmp_dir=None,
            brain_extraction_template=None,
            brain_extraction_prob_mask=None,
            brain_extraction_reg_mask=None,
            anat_template=None,
        )

    def test_parser_from_namespace(self, anat_namespace: argparse.Namespace) -> None:
        """Tests parser successfully validates namespace."""
        args = anatomical.AnatomicalArgs.validate_namespace(anat_namespace)
        assert isinstance(args, anatomical.AnatomicalArgs)

    def test_defaults_use_bundled_templates(
        self, anat_namespace: argparse.Namespace
    ) -> None:
        """When all template args are None, bundled defaults are used."""
        args = anatomical.AnatomicalArgs.validate_namespace(anat_namespace)
        assert args.registration_template == REGISTRATION_TEMPLATES.brain_1mm
        assert (
            args.brain_extraction_templates.template
            == BRAIN_EXTRACTION_TEMPLATES.template
        )
        assert (
            args.brain_extraction_templates.probability_mask
            == BRAIN_EXTRACTION_TEMPLATES.probability_mask
        )
        assert (
            args.brain_extraction_templates.registration_mask
            == BRAIN_EXTRACTION_TEMPLATES.registration_mask
        )
