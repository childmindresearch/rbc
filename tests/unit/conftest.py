"""Fixtures specific to unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pathlib as pl


@pytest.fixture
def test_file(tmp_path: pl.Path) -> pl.Path:
    """Create sample file for testing."""
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("Sample content")
    return test_file
