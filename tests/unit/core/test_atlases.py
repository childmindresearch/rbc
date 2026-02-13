"""Unit tests for rbc.core.metrics.atlases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from rbc.core.metrics.atlases import ATLAS_REGISTRY, get_atlas


class TestGetAtlas:
    """Tests for the get_atlas function."""

    def test_returns_path_for_known_names(self) -> None:
        """get_atlas should return a Path for every registered atlas name."""
        for name in ATLAS_REGISTRY:
            with patch.object(Path, "exists", return_value=True):
                result = get_atlas(name)
            assert isinstance(result, Path)
            assert result.name == ATLAS_REGISTRY[name]

    def test_unknown_name_raises_value_error(self) -> None:
        """Unknown atlas name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown atlas"):
            get_atlas("not_a_real_atlas")  # type: ignore # should raise

    def test_missing_file_raises_file_not_found(self) -> None:
        """Missing file on disk should raise FileNotFoundError."""
        with (
            patch.object(Path, "exists", return_value=False),
            pytest.raises(FileNotFoundError, match="Atlas file not found"),
        ):
            get_atlas("schaefer_200")

    def test_path_ends_with_nii_gz(self) -> None:
        """All atlas paths should point to .nii.gz files."""
        for name in ATLAS_REGISTRY:
            with patch.object(Path, "exists", return_value=True):
                result = get_atlas(name)
            assert result.name.endswith(".nii.gz")
