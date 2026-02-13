"""Unit tests for rbc.core.metrics.standardization."""

from __future__ import annotations

import numpy as np
import pytest

from rbc.core.metrics.standardization import zscore


class TestZscore:
    """Tests for z-score standardization."""

    def test_output_shape(self) -> None:
        """Output shape should match input."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, 10, 10))
        mask = np.ones((10, 10, 10))
        result = zscore(data, mask)
        assert result.shape == data.shape

    def test_output_dtype(self) -> None:
        """Output should be float64."""
        data = np.ones((5, 5, 5), dtype=np.float32)
        mask = np.ones((5, 5, 5))
        # Add variation so std != 0
        data[2, 2, 2] = 2.0
        result = zscore(data, mask)
        assert result.dtype == np.float64

    def test_known_values(self) -> None:
        """Check against manually computed z-scores."""
        data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
        mask = np.ones_like(data)
        result = zscore(data, mask)

        # mean=3, std=sqrt(2)
        expected_mean = 3.0
        expected_std = np.std([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = (data - expected_mean) / expected_std
        np.testing.assert_allclose(result, expected)

    def test_out_of_mask_zero(self) -> None:
        """Voxels outside the mask should be zero."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal((10, 10, 10))
        mask = np.zeros((10, 10, 10))
        mask[3:7, 3:7, 3:7] = 1
        result = zscore(data, mask)

        assert np.all(result[mask == 0] == 0.0)

    def test_constant_data_returns_zeros(self) -> None:
        """Constant in-mask data (std=0) should return all zeros."""
        data = np.ones((8, 8, 8)) * 42.0
        mask = np.ones((8, 8, 8))
        result = zscore(data, mask)
        assert np.all(result == 0.0)

    def test_mean_zero_std_one(self) -> None:
        """Z-scored in-mask values should have mean ~0 and std ~1."""
        rng = np.random.default_rng(2)
        data = rng.standard_normal((20, 20, 20))
        mask = np.ones((20, 20, 20))
        result = zscore(data, mask)

        in_mask = result[mask > 0]
        assert in_mask.mean() == pytest.approx(0.0, abs=1e-10)
        assert in_mask.std() == pytest.approx(1.0, abs=1e-10)

    def test_partial_mask_mean_std(self) -> None:
        """Z-scored in-mask values with partial mask should have mean ~0 and std ~1."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((20, 20, 20))
        mask = np.zeros((20, 20, 20))
        mask[5:15, 5:15, 5:15] = 1
        result = zscore(data, mask)

        in_mask = result[mask > 0]
        assert in_mask.mean() == pytest.approx(0.0, abs=1e-10)
        assert in_mask.std() == pytest.approx(1.0, abs=1e-10)

    def test_rejects_non_3d(self) -> None:
        """Z-score should reject non-3D input."""
        with pytest.raises(ValueError, match="3D"):
            zscore(np.zeros((5, 5)), np.ones((5, 5)))

        with pytest.raises(ValueError, match="3D"):
            zscore(np.zeros((5, 5, 5, 10)), np.ones((5, 5, 5, 10)))

    def test_float_mask(self) -> None:
        """Float mask with values > 0 should be treated as True."""
        rng = np.random.default_rng(4)
        data = rng.standard_normal((8, 8, 8))
        bool_mask = np.ones((8, 8, 8))
        float_mask = np.ones((8, 8, 8)) * 0.3

        result_bool = zscore(data, bool_mask)
        result_float = zscore(data, float_mask)

        np.testing.assert_array_equal(result_bool, result_float)

    def test_deterministic(self) -> None:
        """Same input should produce identical output."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal((8, 8, 8))
        mask = np.ones((8, 8, 8))

        r1 = zscore(data, mask)
        r2 = zscore(data, mask)

        np.testing.assert_array_equal(r1, r2)

    def test_negative_mask_values_excluded(self) -> None:
        """Negative mask values should be treated as outside the mask."""
        rng = np.random.default_rng(6)
        data = rng.standard_normal((8, 8, 8))
        mask = np.ones((8, 8, 8))
        mask[:4] = -1

        result = zscore(data, mask)
        assert np.all(result[:4] == 0.0)
