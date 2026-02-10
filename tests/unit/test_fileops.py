"""Unit tests for file operation helpers."""

from __future__ import annotations

import pathlib as pl
import tempfile

import pytest

from rbc.core import fileops


class TestFileTmpCopy:
    """Test suite for fileops.file_tmp_copy."""

    def test_copy_successful(self, test_file: pl.Path) -> None:
        """Test copy created successfully."""
        with fileops.file_tmp_copy(test_file) as tmp_file:
            assert isinstance(tmp_file, pl.Path)
            assert tmp_file.exists()
            assert tmp_file.is_file()
            assert tmp_file.read_text() == "Sample content"

    def test_copy_correct_name(self, test_file: pl.Path) -> None:
        """Test copy has same name as original."""
        with fileops.file_tmp_copy(test_file) as tmp_file:
            assert tmp_file.name == test_file.name

    def test_copy_in_temp_dir(self, test_file: pl.Path) -> None:
        """Test copy was created in temporary directory."""
        with fileops.file_tmp_copy(test_file) as tmp_file:
            assert str(tmp_file.parent).startswith(tempfile.gettempdir())

    def test_cleanup(self, test_file: pl.Path) -> None:
        """Test successful cleanup after normal exit."""
        with fileops.file_tmp_copy(test_file) as tmp_file:
            tmp_dir = tmp_file.parent
            assert tmp_dir.exists()
        assert not tmp_file.exists()
        assert not tmp_dir.exists()


class TestFileRename:
    """Test suite for fileops.file_rename."""

    def test_rename_successful(self, test_file: pl.Path) -> None:
        """Test file renamed successfully."""
        new_path = fileops.file_rename(test_file, "renamed.txt")
        assert isinstance(new_path, pl.Path)
        assert new_path.exists()
        assert new_path.is_file()
        assert new_path.name == "renamed.txt"
        assert not test_file.exists()

    def test_content_preserved(self, test_file: pl.Path) -> None:
        """Test content preserved in renamed file."""
        new_path = fileops.file_rename(test_file, "renamed.txt")
        assert new_path.read_text() == "Sample content"

    def test_same_directory(self, test_file: pl.Path) -> None:
        """Test renamed file stays in same directory."""
        orig_parent = test_file.parent
        new_path = fileops.file_rename(test_file, "renamed.txt")
        assert new_path.parent == orig_parent

    def test_str_input(self, test_file: pl.Path) -> None:
        """Test method accepts string paths."""
        new_path = fileops.file_rename(str(test_file), "renamed.txt")
        assert isinstance(new_path, pl.Path)
        assert new_path.exists()

    def test_rename_existing_fails(self, test_file: pl.Path) -> None:
        """Test renaming to existing file name raises error."""
        existing = test_file.parent / "existing.txt"
        existing.touch()
        with pytest.raises(FileExistsError, match="already exists"):
            fileops.file_rename(test_file, "existing.txt")

    def test_nonexistent_file_fails(self, tmp_path: pl.Path) -> None:
        """Test renaming nonexistent file raises error."""
        non_existent = tmp_path / "fake_file.txt"
        with pytest.raises(FileNotFoundError):
            fileops.file_rename(non_existent, "new_name.txt")
