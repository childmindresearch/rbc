"""Unit tests for utility methods."""

import pathlib as pl
import tempfile
from types import SimpleNamespace

import pytest

from rbc.core import utils


class TestCreateCopy:
    """Test suite for utils.create_copy."""

    @pytest.fixture
    def test_file(self, tmp_path: pl.Path) -> pl.Path:
        """Create sample file for testing."""
        test_file = tmp_path / "test_file.txt"
        test_file.write_text("Sample content")
        return test_file

    def test_copy_successful(self, test_file: pl.Path) -> None:
        """Test copy created successfully."""
        with utils.create_copy(test_file) as tmp_file:
            assert isinstance(tmp_file, pl.Path)
            assert tmp_file.exists() and tmp_file.is_file()
            assert tmp_file.read_text() == "Sample content"

    def test_copy_correct_name(self, test_file: pl.Path) -> None:
        """Test copy has same name as original."""
        with utils.create_copy(test_file) as tmp_file:
            assert tmp_file.name == test_file.name

    def test_copy_in_temp_dir(self, test_file: pl.Path) -> None:
        """Test copy was created in temporary directory."""
        with utils.create_copy(test_file) as tmp_file:
            assert str(tmp_file.parent).startswith(tempfile.gettempdir())

    def test_cleanup(self, test_file: pl.Path) -> None:
        """Test successful cleanup after normal exit."""
        with utils.create_copy(test_file) as tmp_file:
            tmp_dir = tmp_file.parent
            assert tmp_dir.exists()
        assert not tmp_file.exists() and not tmp_dir.exists()


class TestGetBaseEntities:
    """Test suite for utils.get_base_entities."""

    def test_default_entities_extracted(self, test_subject: SimpleNamespace) -> None:
        """Test default base entities, if they exist are extracted."""
        result = utils.get_base_entities(test_subject.bold)
        assert isinstance(result, dict)
        assert "ses" not in result.keys()
        assert result == {"sub": "01", "run": "01"}

    def test_custom_entities_extracted(self, test_subject: SimpleNamespace) -> None:
        """Test custom base entities are extracted if exists."""
        result = utils.get_base_entities(test_subject.bold, ["sub", "task"])
        assert result == {"sub": "01", "task": "balloonanalogrisktask"}

    def test_nonexistent_entities(self, test_subject: SimpleNamespace) -> None:
        """Test nonexistent entities are not returned."""
        result = utils.get_base_entities(test_subject.bold, base_entities=["acq", "ce"])
        assert result == {}
