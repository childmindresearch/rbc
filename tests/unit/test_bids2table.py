"""Unit tests for Niwrap helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rbc.core.bids2table import load_table

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
        with (
            patch("bids2table.batch_index_dataset") as mock_batch,
            patch("bids2table.find_bids_datasets") as mock_find,
        ):
            mock_find.return_value = [tmp_path]
            mock_batch.return_value = [test_table]
            result = load_table(
                dataset_dir=tmp_path, index_fpath=tmp_path / "index.parquet"
            )
            mock_find.assert_called_once()
            mock_batch.assert_called_once()
            assert isinstance(result, pl.DataFrame)

    def test_load_without_index(self, tmp_path: Path, test_table: pa.Table) -> None:
        """Test indexing without passing an index."""
        with (
            patch("bids2table.batch_index_dataset") as mock_batch,
            patch("bids2table.find_bids_datasets") as mock_find,
        ):
            mock_find.return_value = [tmp_path]
            mock_batch.return_value = [test_table]
            result = load_table(dataset_dir=tmp_path)
            mock_find.assert_called_once()
            mock_batch.assert_called_once()
            assert isinstance(result, pl.DataFrame)
