"""Unit tests for Niwrap helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rbc.core.bids2table import get_extra_entity, load_table

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadBidsTable:
    """Testing suite for bids2table.load_table."""

    @pytest.fixture
    def test_table(self) -> pa.Table:
        """Create Arrow table for testing."""
        return pa.table(
            {
                "sub": ["sub-01", "sub-01", "sub-02"],
                "ses": ["ses-01", "ses-02", "ses-01"],
            }
        )

    def test_load_existing_index(self, tmp_path: Path, test_table: pa.Table) -> None:
        """Test loading from existing parquet file."""
        index_path = tmp_path / "index.parquet"
        pq.write_table(test_table, index_path)
        result = load_table(dataset_dir=tmp_path, index_fpath=index_path)
        assert isinstance(result, pl.DataFrame)

    def test_load_no_existing_index(self, tmp_path: Path, test_table: pa.Table) -> None:
        """Testing indexing if existing index does not exist."""
        index_fpath = tmp_path / "index.parquet"
        with (
            patch("rbc.core.bids2table.b2t.batch_index_dataset") as mock_batch,
            patch("rbc.core.bids2table.b2t.find_bids_datasets") as mock_find,
        ):
            mock_find.return_value = [tmp_path]
            mock_batch.return_value = [test_table]
            result = load_table(dataset_dir=tmp_path, index_fpath=index_fpath)
            mock_find.assert_called_once()
            mock_batch.assert_called_once()
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 2)
            assert index_fpath.exists()

    def test_load_without_index(self, tmp_path: Path, test_table: pa.Table) -> None:
        """Test indexing without passing an index."""
        with (
            patch("rbc.core.bids2table.b2t.batch_index_dataset") as mock_batch,
            patch("rbc.core.bids2table.b2t.find_bids_datasets") as mock_find,
        ):
            mock_find.return_value = [tmp_path]
            mock_batch.return_value = [test_table]
            result = load_table(dataset_dir=tmp_path)
            mock_find.assert_called_once()
            mock_batch.assert_called_once()
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 2)


class TestGetExtraEntity:
    """Testing suite for bids2table.get_extra_entity."""

    @pytest.fixture
    def test_table(self) -> pl.DataFrame:
        """DataFrame for testing extra_entities."""
        return pl.DataFrame(
            {
                "subject": ["sub-01", "sub-02"],
                "extra_entities": [
                    [
                        {"key": "foo", "value": "bar"},
                        {"key": "acq", "value": "multiband"},
                    ],
                    [{"key": "foo", "value": "bar"}],
                ],
            }
        )

    def test_key_exists(self, test_table: pl.DataFrame) -> None:
        """Test existing key returns value."""
        result = test_table.with_columns(foo=get_extra_entity("foo"))
        assert result["foo"].to_list() == ["bar", "bar"]

    def test_missing_row_return_none(self, test_table: pl.DataFrame) -> None:
        """Test None is returned if entity missing."""
        result = test_table.with_columns(acq=get_extra_entity("acq"))
        assert result["acq"].to_list() == ["multiband", None]

    def test_non_existent_key(self, test_table: pl.DataFrame) -> None:
        """Test None is returned for all rows if key is non-existent."""
        result = test_table.with_columns(missing=get_extra_entity("missing"))
        assert result["missing"].to_list() == [None, None]
