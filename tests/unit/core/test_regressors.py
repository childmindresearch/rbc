"""Unit tests for rbc.core.functional.regressors."""

from __future__ import annotations

import numpy as np
import pytest

from rbc.core.functional.regressors import (
    assemble_36param_regressors,
    assemble_acompcor_regressors,
    check_regressor_rank,
    compute_acompcor,
    compute_delayed,
    expand_motion_params,
    expand_regressor,
    extract_mean_signal,
)

T = 50  # timepoints


# ===================================================================
# compute_delayed
# ===================================================================
class TestComputeDelayed:
    """Tests for lag-1 delayed signal."""

    def test_shape(self) -> None:
        """Output shape should match input."""
        signal = np.random.default_rng(0).standard_normal(T)
        result = compute_delayed(signal)
        assert result.shape == (T,)

    def test_first_element_zero(self) -> None:
        """First element should be zero."""
        signal = np.random.default_rng(1).standard_normal(T)
        result = compute_delayed(signal)
        assert result[0] == 0.0

    def test_known_values(self) -> None:
        """Known input should produce lag-1 shifted copy."""
        signal = np.array([1.0, 3.0, 6.0, 10.0])
        result = compute_delayed(signal)
        np.testing.assert_array_equal(result, [0.0, 1.0, 3.0, 6.0])

    def test_constant_signal(self) -> None:
        """Constant signal delayed should be constant (except first)."""
        signal = np.ones(T) * 5.0
        result = compute_delayed(signal)
        expected = np.ones(T) * 5.0
        expected[0] = 0.0
        np.testing.assert_array_equal(result, expected)

    def test_rejects_non_1d(self) -> None:
        """Non-1D input should raise."""
        with pytest.raises(ValueError, match="1D"):
            compute_delayed(np.zeros((5, 3)))


# ===================================================================
# expand_regressor
# ===================================================================
class TestExpandRegressor:
    """Tests for 1-to-4 regressor expansion."""

    def test_shape(self) -> None:
        """Output should be (T, 4)."""
        signal = np.random.default_rng(2).standard_normal(T)
        result = expand_regressor(signal)
        assert result.shape == (T, 4)

    def test_columns_are_correct(self) -> None:
        """Columns should be [original, delayed, squared, delayed_squared]."""
        signal = np.array([1.0, 3.0, 6.0, 10.0])
        result = expand_regressor(signal)
        delayed = compute_delayed(signal)

        np.testing.assert_array_equal(result[:, 0], signal)
        np.testing.assert_array_equal(result[:, 1], delayed)
        np.testing.assert_array_equal(result[:, 2], signal**2)
        np.testing.assert_array_equal(result[:, 3], delayed**2)

    def test_dtype_float64(self) -> None:
        """Output dtype should be float64."""
        signal = np.arange(T, dtype=np.int32)
        result = expand_regressor(signal.astype(np.float64))
        assert result.dtype == np.float64

    def test_rejects_non_1d(self) -> None:
        """Non-1D input should raise."""
        with pytest.raises(ValueError, match="1D"):
            expand_regressor(np.zeros((5, 3)))


# ===================================================================
# expand_motion_params
# ===================================================================
class TestExpandMotionParams:
    """Tests for 6-to-24 motion parameter expansion."""

    def test_shape(self) -> None:
        """Output should be (T, 24)."""
        params = np.random.default_rng(3).standard_normal((T, 6))
        result = expand_motion_params(params)
        assert result.shape == (T, 24)

    def test_dtype_float64(self) -> None:
        """Output dtype should be float64."""
        params = np.random.default_rng(4).standard_normal((T, 6))
        result = expand_motion_params(params)
        assert result.dtype == np.float64

    def test_rejects_wrong_columns(self) -> None:
        """Input with != 6 columns should raise."""
        with pytest.raises(ValueError, match="6"):
            expand_motion_params(np.zeros((T, 3)))
        with pytest.raises(ValueError, match="6"):
            expand_motion_params(np.zeros((T,)))


# ===================================================================
# extract_mean_signal
# ===================================================================
class TestExtractMeanSignal:
    """Tests for mean signal extraction from masked voxels."""

    def test_shape(self) -> None:
        """Output should be 1-D of length T."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal((4, 5, 6, T))
        mask = np.ones((4, 5, 6))
        result = extract_mean_signal(data, mask)
        assert result.shape == (T,)

    def test_full_mask_equals_global_mean(self) -> None:
        """Full mask should give global mean timeseries."""
        rng = np.random.default_rng(6)
        data = rng.standard_normal((4, 5, 6, T))
        mask = np.ones((4, 5, 6))
        result = extract_mean_signal(data, mask)
        expected = data.reshape(-1, T).mean(axis=0)
        np.testing.assert_allclose(result, expected)

    def test_single_voxel(self) -> None:
        """Single-voxel mask should return that voxel's timeseries."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((4, 5, 6, T))
        mask = np.zeros((4, 5, 6))
        mask[2, 3, 4] = 1
        result = extract_mean_signal(data, mask)
        np.testing.assert_allclose(result, data[2, 3, 4, :])

    def test_rejects_empty_mask(self) -> None:
        """Empty mask should raise."""
        data = np.zeros((4, 5, 6, T))
        mask = np.zeros((4, 5, 6))
        with pytest.raises(ValueError, match="empty"):
            extract_mean_signal(data, mask)

    def test_rejects_non_4d_data(self) -> None:
        """Non-4D data should raise."""
        with pytest.raises(ValueError, match="4D"):
            extract_mean_signal(np.zeros((4, 5, 6)), np.ones((4, 5, 6)))

    def test_rejects_non_3d_mask(self) -> None:
        """Non-3D mask should raise."""
        with pytest.raises(ValueError, match="3D"):
            extract_mean_signal(np.zeros((4, 5, 6, T)), np.ones((4, 5)))

    def test_rejects_shape_mismatch(self) -> None:
        """Spatial dimension mismatch should raise."""
        with pytest.raises(ValueError, match="Spatial dimensions"):
            extract_mean_signal(np.zeros((4, 5, 6, T)), np.ones((4, 5, 7)))


# ===================================================================
# compute_acompcor
# ===================================================================
class TestComputeAcompcor:
    """Tests for aCompCor PCA computation."""

    def test_shape(self) -> None:
        """Output should be (T, n_components)."""
        rng = np.random.default_rng(10)
        data = rng.standard_normal((6, 6, 6, T))
        mask = np.ones((6, 6, 6))
        result = compute_acompcor(data, mask, n_components=5)
        assert result.shape == (T, 5)

    def test_orthogonal_components(self) -> None:
        """Components should be approximately orthogonal."""
        rng = np.random.default_rng(11)
        data = rng.standard_normal((6, 6, 6, T))
        mask = np.ones((6, 6, 6))
        result = compute_acompcor(data, mask, n_components=5)
        gram = result.T @ result
        off_diagonal = gram - np.diag(np.diag(gram))
        assert np.allclose(off_diagonal, 0, atol=1e-10)

    def test_rejects_too_few_voxels(self) -> None:
        """Fewer voxels than components should raise."""
        data = np.random.default_rng(12).standard_normal((3, 3, 3, T))
        mask = np.zeros((3, 3, 3))
        mask[0, 0, 0] = 1
        mask[0, 0, 1] = 1
        with pytest.raises(ValueError, match="Too few voxels"):
            compute_acompcor(data, mask, n_components=5)

    def test_dtype_float64(self) -> None:
        """Output dtype should be float64."""
        rng = np.random.default_rng(13)
        data = rng.standard_normal((6, 6, 6, T))
        mask = np.ones((6, 6, 6))
        result = compute_acompcor(data, mask)
        assert result.dtype == np.float64

    def test_rejects_non_4d_data(self) -> None:
        """Non-4D data should raise."""
        with pytest.raises(ValueError, match="4D"):
            compute_acompcor(np.zeros((5, 5, 5)), np.ones((5, 5, 5)))

    def test_rejects_non_3d_mask(self) -> None:
        """Non-3D mask should raise."""
        with pytest.raises(ValueError, match="3D"):
            compute_acompcor(np.zeros((5, 5, 5, T)), np.ones((5, 5)))


# ===================================================================
# assemble_36param_regressors
# ===================================================================
class TestAssemble36ParamRegressors:
    """Tests for 36-parameter regressor assembly."""

    def test_shape(self) -> None:
        """Output matrix should be (T, 36)."""
        rng = np.random.default_rng(20)
        motion = rng.standard_normal((T, 6))
        csf = rng.standard_normal(T)
        wm = rng.standard_normal(T)
        gs = rng.standard_normal(T)
        matrix, _ = assemble_36param_regressors(motion, csf, wm, gs)
        assert matrix.shape == (T, 36)

    def test_column_names_count(self) -> None:
        """Should have exactly 36 unique column names."""
        rng = np.random.default_rng(21)
        motion = rng.standard_normal((T, 6))
        csf = rng.standard_normal(T)
        wm = rng.standard_normal(T)
        gs = rng.standard_normal(T)
        _, names = assemble_36param_regressors(motion, csf, wm, gs)
        assert len(names) == 36
        assert len(set(names)) == 36


# ===================================================================
# assemble_acompcor_regressors
# ===================================================================
class TestAssembleAcompcorRegressors:
    """Tests for aCompCor regressor assembly."""

    def test_shape(self) -> None:
        """Output matrix should be (T, 37)."""
        rng = np.random.default_rng(30)
        motion = rng.standard_normal((T, 6))
        csf = rng.standard_normal(T)
        wm = rng.standard_normal(T)
        acompcor = rng.standard_normal((T, 5))
        matrix, _ = assemble_acompcor_regressors(motion, csf, wm, acompcor)
        assert matrix.shape == (T, 37)

    def test_column_names_count(self) -> None:
        """Should have exactly 37 unique column names."""
        rng = np.random.default_rng(31)
        motion = rng.standard_normal((T, 6))
        csf = rng.standard_normal(T)
        wm = rng.standard_normal(T)
        acompcor = rng.standard_normal((T, 5))
        _, names = assemble_acompcor_regressors(motion, csf, wm, acompcor)
        assert len(names) == 37
        assert len(set(names)) == 37


# ===================================================================
# check_regressor_rank
# ===================================================================
class TestCheckRegressorRank:
    """Tests for regressor conditioning check."""

    def test_full_rank_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Full-rank matrix should not produce a warning."""
        rng = np.random.default_rng(40)
        matrix = rng.standard_normal((T, 10))
        names = [f"col_{i}" for i in range(10)]
        with caplog.at_level("WARNING"):
            check_regressor_rank(matrix, names)
        assert "rank-deficient" not in caplog.text

    def test_rank_deficient_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Rank-deficient matrix should log a warning."""
        rng = np.random.default_rng(41)
        col = rng.standard_normal(T)
        # Duplicate column makes the matrix rank-deficient
        matrix = np.column_stack([col, col, rng.standard_normal((T, 3))])
        names = ["a", "a_dup", "b", "c", "d"]
        with caplog.at_level("WARNING"):
            check_regressor_rank(matrix, names)
        assert "rank-deficient" in caplog.text
