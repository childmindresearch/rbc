"""Fixtures specific to unit tests."""

from __future__ import annotations

import pathlib as pl

import pytest


@pytest.fixture
def test_file(tmp_path: pl.Path) -> pl.Path:
    """Create sample file for testing."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("Sample content")
    return test_file
