"""Unit tests for utility methods."""

import pathlib as pl
import tempfile

from rbc.core import utils


class TestCreateCopy:
    """Test suite for utils.create_copy."""

    def test_copy_successful(self, test_file: pl.Path) -> None:
        """Test copy created successfully."""
        with utils.create_copy(test_file) as tmp_file:
            assert isinstance(tmp_file, pl.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
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
        assert not tmp_file.exists()
        assert not tmp_dir.exists()
