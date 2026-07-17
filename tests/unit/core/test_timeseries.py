"""Unit tests for rbc.core.metrics.timeseries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from rbc.core.metrics.timeseries import (
    compute_timeseries,
    correlation_matrix,
    extract_timeseries,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestExtractTimeseries:
    """Tests for extract_timeseries."""

    def test_output_shape(self) -> None:
        """Output should be (n_rois, n_timepoints)."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((4, 5, 6, 20))
        atlas = np.zeros((4, 5, 6), dtype=int)
        atlas[0:2, :, :] = 1
        atlas[2:4, :, :] = 2

        result = extract_timeseries(data, atlas)
        assert result.shape == (2, 20)
        assert result.dtype == np.float64

    def test_single_roi(self) -> None:
        """A single ROI should produce shape (1, T)."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal((3, 3, 3, 10))
        atlas = np.ones((3, 3, 3), dtype=int)

        result = extract_timeseries(data, atlas)
        assert result.shape == (1, 10)

    def test_mean_with_known_values(self) -> None:
        """Mean should match hand-computed values."""
        data = np.zeros((2, 2, 1, 3))
        data[0, 0, 0, :] = [1.0, 2.0, 3.0]
        data[0, 1, 0, :] = [3.0, 4.0, 5.0]
        data[1, 0, 0, :] = [10.0, 20.0, 30.0]
        data[1, 1, 0, :] = [20.0, 40.0, 60.0]

        atlas = np.array([[[1], [1]], [[2], [2]]])
        result = extract_timeseries(data, atlas)

        # ROI 1: mean of (1,2,3) and (3,4,5) = (2,3,4)
        np.testing.assert_allclose(result[0], [2.0, 3.0, 4.0])
        # ROI 2: mean of (10,20,30) and (20,40,60) = (15,30,45)
        np.testing.assert_allclose(result[1], [15.0, 30.0, 45.0])

    def test_background_exclusion(self) -> None:
        """Label 0 should be treated as background and excluded."""
        data = np.ones((3, 3, 3, 5))
        atlas = np.zeros((3, 3, 3), dtype=int)
        atlas[1, 1, 1] = 1

        result = extract_timeseries(data, atlas)
        assert result.shape == (1, 5)

    def test_custom_labels(self) -> None:
        """Custom labels should select a subset of ROIs."""
        rng = np.random.default_rng(2)
        data = rng.standard_normal((4, 4, 4, 10))
        atlas = np.zeros((4, 4, 4), dtype=int)
        atlas[0:2, :, :] = 1
        atlas[2:3, :, :] = 2
        atlas[3:4, :, :] = 3

        result = extract_timeseries(data, atlas, labels=np.array([1, 3]))
        assert result.shape == (2, 10)

    def test_label_ordering(self) -> None:
        """Output rows should follow the order of the labels array."""
        data = np.zeros((2, 1, 1, 3))
        data[0, 0, 0, :] = [1.0, 2.0, 3.0]
        data[1, 0, 0, :] = [10.0, 20.0, 30.0]

        atlas = np.array([[[1]], [[2]]])

        # Reversed order
        result = extract_timeseries(data, atlas, labels=np.array([2, 1]))
        np.testing.assert_allclose(result[0], [10.0, 20.0, 30.0])
        np.testing.assert_allclose(result[1], [1.0, 2.0, 3.0])

    def test_non_contiguous_labels(self) -> None:
        """Non-contiguous label values should work (e.g. 5, 10)."""
        data = np.ones((4, 4, 4, 5))
        atlas = np.zeros((4, 4, 4), dtype=int)
        atlas[0:2, :, :] = 5
        atlas[2:4, :, :] = 10

        result = extract_timeseries(data, atlas)
        assert result.shape == (2, 5)

    def test_rejects_non_4d_data(self) -> None:
        """Should reject non-4D data."""
        with pytest.raises(ValueError, match="4D"):
            extract_timeseries(np.zeros((5, 5, 5)), np.ones((5, 5, 5), dtype=int))

    def test_rejects_non_3d_atlas(self) -> None:
        """Should reject non-3D atlas."""
        with pytest.raises(ValueError, match="3D"):
            extract_timeseries(np.zeros((5, 5, 5, 10)), np.ones((5, 5), dtype=int))

    def test_rejects_shape_mismatch(self) -> None:
        """Should reject spatial dimension mismatch."""
        with pytest.raises(ValueError, match="Spatial dimensions"):
            extract_timeseries(
                np.zeros((4, 5, 6, 10)),
                np.ones((4, 5, 7), dtype=int),
            )

    def test_rejects_empty_atlas(self) -> None:
        """Should reject atlas with only background (label 0)."""
        with pytest.raises(ValueError, match="No ROI labels"):
            extract_timeseries(
                np.zeros((4, 4, 4, 10)),
                np.zeros((4, 4, 4), dtype=int),
            )

    def test_deterministic(self) -> None:
        """Same input should produce identical output."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((4, 4, 4, 10))
        atlas = np.zeros((4, 4, 4), dtype=int)
        atlas[0:2, :, :] = 1
        atlas[2:4, :, :] = 2

        r1 = extract_timeseries(data, atlas)
        r2 = extract_timeseries(data, atlas)
        np.testing.assert_array_equal(r1, r2)


class TestCorrelationMatrix:
    """Tests for correlation_matrix."""

    def test_shape_and_symmetry(self) -> None:
        """Output should be (n_rois, n_rois) and symmetric."""
        rng = np.random.default_rng(10)
        ts = rng.standard_normal((5, 20))
        corr = correlation_matrix(ts)

        assert corr.shape == (5, 5)
        np.testing.assert_allclose(corr, corr.T)

    def test_diagonal_is_one(self) -> None:
        """Diagonal elements should be 1."""
        rng = np.random.default_rng(11)
        ts = rng.standard_normal((4, 30))
        corr = correlation_matrix(ts)

        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_identical_timeseries(self) -> None:
        """Identical timeseries should yield all-ones matrix."""
        ts_single = np.arange(20, dtype=np.float64)
        ts = np.tile(ts_single, (3, 1))
        corr = correlation_matrix(ts)

        np.testing.assert_allclose(corr, np.ones((3, 3)))

    def test_anticorrelated(self) -> None:
        """Perfectly anticorrelated timeseries should yield -1."""
        ts = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [-1.0, -2.0, -3.0, -4.0, -5.0],
            ]
        )
        corr = correlation_matrix(ts)

        assert corr[0, 1] == pytest.approx(-1.0)
        assert corr[1, 0] == pytest.approx(-1.0)

    def test_bounded(self) -> None:
        """All values should be in [-1, 1]."""
        rng = np.random.default_rng(12)
        ts = rng.standard_normal((10, 50))
        corr = correlation_matrix(ts)

        assert corr.min() >= -1.0 - 1e-10
        assert corr.max() <= 1.0 + 1e-10

    def test_zero_variance_roi_produces_nan(self) -> None:
        """Zero-variance ROIs should yield NaN without RuntimeWarning."""
        ts = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [7.0, 7.0, 7.0, 7.0, 7.0],  # constant
                [5.0, 4.0, 3.0, 2.0, 1.0],
            ]
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            corr = correlation_matrix(ts)

        # Valid ROIs should have real correlation values
        assert corr[0, 2] == pytest.approx(-1.0)
        # Zero-variance ROI should produce NaN
        assert np.isnan(corr[1, 0])
        assert np.isnan(corr[0, 1])
        assert np.isnan(corr[1, 2])
        assert corr[1, 1] == pytest.approx(1.0)

    def test_all_zero_variance_produces_all_nan(self) -> None:
        """If all ROIs are constant, the entire matrix should be NaN."""
        ts = np.array(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
            ]
        )
        corr = correlation_matrix(ts)
        assert np.all(np.isnan(corr[~np.eye(2, dtype=bool)]))
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_rejects_single_roi(self) -> None:
        """Should reject fewer than 2 ROIs."""
        with pytest.raises(ValueError, match="2 ROIs"):
            correlation_matrix(np.array([[1.0, 2.0, 3.0]]))

    def test_rejects_single_timepoint(self) -> None:
        """Should reject fewer than 2 timepoints."""
        with pytest.raises(ValueError, match="2 timepoints"):
            correlation_matrix(np.array([[1.0], [2.0]]))


class TestComputeTimeseries:
    """Tests for the file I/O wrapper compute_timeseries."""

    def _make_nifti(self, data: np.ndarray, path: Path) -> None:
        """Write a synthetic NIfTI file."""
        import nibabel as nib

        img = nib.nifti1.Nifti1Image(data, affine=np.eye(4))
        img.to_filename(str(path))

    def test_round_trip(self, tmp_path: Path) -> None:
        """Should produce Parquet files that can be loaded back."""
        rng = np.random.default_rng(20)
        data = rng.standard_normal((4, 4, 4, 10))
        atlas = np.zeros((4, 4, 4), dtype=np.int16)
        atlas[0:2, :, :] = 1
        atlas[2:4, :, :] = 2

        in_file = tmp_path / "bold.nii.gz"
        atlas_file = tmp_path / "atlas.nii.gz"
        self._make_nifti(data, in_file)
        self._make_nifti(atlas.astype(np.float64), atlas_file)

        result = compute_timeseries(in_file, atlas_file)

        assert result.timeseries.exists()
        assert result.connectome.exists()
        assert len(result.labels) == 2

        ts_loaded = pl.read_parquet(result.timeseries)
        assert ts_loaded.shape == (2, 10)

        corr_loaded = pl.read_parquet(result.connectome)
        assert corr_loaded.shape == (2, 2)

    def test_output_naming(self, tmp_path: Path) -> None:
        """Output files should follow <stem>_timeseries.tsv naming."""
        rng = np.random.default_rng(21)
        data = rng.standard_normal((3, 3, 3, 5))
        atlas = np.ones((3, 3, 3), dtype=np.int16)
        atlas[0, 0, 0] = 2  # need >=2 ROIs for correlation

        in_file = tmp_path / "sub-01_bold.nii.gz"
        atlas_file = tmp_path / "atlas.nii.gz"
        self._make_nifti(data, in_file)
        self._make_nifti(atlas.astype(np.float64), atlas_file)

        result = compute_timeseries(in_file, atlas_file)

        assert result.timeseries.name == "sub-01_bold_timeseries.parquet"
        assert result.connectome.name == "sub-01_bold_connectome.parquet"

    def test_custom_out_dir(self, tmp_path: Path) -> None:
        """Should write to a custom output directory."""
        rng = np.random.default_rng(22)
        data = rng.standard_normal((3, 3, 3, 5))
        atlas = np.ones((3, 3, 3), dtype=np.int16)
        atlas[0, 0, 0] = 2

        in_file = tmp_path / "bold.nii.gz"
        atlas_file = tmp_path / "atlas.nii.gz"
        out_dir = tmp_path / "output"
        self._make_nifti(data, in_file)
        self._make_nifti(atlas.astype(np.float64), atlas_file)

        result = compute_timeseries(in_file, atlas_file, out_dir=out_dir)

        assert result.timeseries.parent == out_dir
        assert result.connectome.parent == out_dir

    def test_labels_in_output(self, tmp_path: Path) -> None:
        """Output labels should match the atlas ROI labels."""
        data = np.ones((4, 4, 4, 5))
        atlas = np.zeros((4, 4, 4), dtype=np.int16)
        atlas[0:2, :, :] = 3
        atlas[2:4, :, :] = 7

        in_file = tmp_path / "bold.nii.gz"
        atlas_file = tmp_path / "atlas.nii.gz"
        self._make_nifti(data, in_file)
        self._make_nifti(atlas.astype(np.float64), atlas_file)

        result = compute_timeseries(in_file, atlas_file)
        np.testing.assert_array_equal(result.labels, [3, 7])
