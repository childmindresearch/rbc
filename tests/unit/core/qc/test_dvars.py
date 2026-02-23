"""Unit tests for rbc.core.qc.dvars."""

from __future__ import annotations

import numpy as np
import pytest

from rbc.core.qc.dvars import (
    DVARSQCMetrics,
    dvars,
    dvars_qc_metrics,
    motion_dvars_correlation,
)

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------
SHAPE = (3, 3, 3, 20)  # (X, Y, Z, T)


def _full_mask(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    """Return an all-ones brain mask matching spatial dims of *shape*."""
    return np.ones(shape[:3])


# ===================================================================
# dvars
# ===================================================================
class TestDvars:
    """Tests for DVARS computation from 4-D data."""

    def test_constant_signal(self) -> None:
        """Constant timeseries → DVARS all zeros."""
        data = np.ones(SHAPE)
        dv = dvars(data, _full_mask())
        np.testing.assert_array_equal(dv, np.zeros(SHAPE[3]))

    def test_output_length(self) -> None:
        """Output length equals number of volumes."""
        data = np.random.default_rng(0).standard_normal(SHAPE)
        dv = dvars(data, _full_mask())
        assert len(dv) == SHAPE[3]

    def test_first_value_is_zero(self) -> None:
        """First DVARS value is always 0."""
        data = np.random.default_rng(1).standard_normal(SHAPE)
        dv = dvars(data, _full_mask())
        assert dv[0] == 0.0

    def test_nonnegative(self) -> None:
        """DVARS should always be non-negative."""
        data = np.random.default_rng(2).standard_normal(SHAPE)
        dv = dvars(data, _full_mask())
        assert np.all(dv >= 0)

    def test_known_step_change(self) -> None:
        """Single step change of known magnitude → known DVARS at that point."""
        data = np.zeros((1, 1, 1, 5))
        # Volume 2 jumps by 3.0 relative to volume 1
        data[0, 0, 0, :] = [0.0, 0.0, 3.0, 3.0, 3.0]
        mask = np.ones((1, 1, 1))
        dv = dvars(data, mask)
        # diff = [0, 3, 0, 0] → DVARS = [0, 0, 3, 0, 0] (with prepended 0)
        np.testing.assert_allclose(dv, [0.0, 0.0, 3.0, 0.0, 0.0])

    def test_respects_mask(self) -> None:
        """Out-of-mask voxels should not contribute to DVARS."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((3, 3, 3, 10))
        # Mask only the centre voxel
        mask = np.zeros((3, 3, 3))
        mask[1, 1, 1] = 1

        dv_masked = dvars(data, mask)
        # Compare against manual single-voxel computation
        single_voxel = data[1, 1, 1, :]
        diff = np.diff(single_voxel)
        expected = np.insert(np.abs(diff), 0, 0.0)
        np.testing.assert_allclose(dv_masked, expected)

    def test_empty_mask(self) -> None:
        """All-zero mask → DVARS should be NaN or zero-filled.

        With no voxels, mean over an empty axis gives NaN. We verify
        the function does not crash and the first value is still 0.
        """
        data = np.random.default_rng(4).standard_normal(SHAPE)
        mask = np.zeros(SHAPE[:3])
        dv = dvars(data, mask)
        assert dv[0] == 0.0
        assert len(dv) == SHAPE[3]

    def test_rejects_non_4d(self) -> None:
        """Non-4-D data should raise ValueError."""
        with pytest.raises(ValueError, match="4D"):
            dvars(np.zeros((5, 5, 5)), np.ones((5, 5, 5)))

    def test_deterministic(self) -> None:
        """Same input → identical output."""
        data = np.random.default_rng(5).standard_normal(SHAPE)
        mask = _full_mask()
        dv1 = dvars(data, mask)
        dv2 = dvars(data, mask)
        np.testing.assert_array_equal(dv1, dv2)

    def test_linear_ramp_uniform(self) -> None:
        """Uniform voxels with linear ramp → constant DVARS after first."""
        # Every voxel increases by 2.0 each volume
        data = np.zeros((2, 2, 2, 10))
        for t in range(10):
            data[:, :, :, t] = t * 2.0
        mask = np.ones((2, 2, 2))
        dv = dvars(data, mask)
        assert dv[0] == 0.0
        np.testing.assert_allclose(dv[1:], 2.0)


# ===================================================================
# motion_dvars_correlation
# ===================================================================
class TestMotionDvarsCorrelation:
    """Tests for Pearson correlation between DVARS and FD."""

    def test_perfect_positive(self) -> None:
        """Identical timeseries → correlation = 1."""
        ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        corr = motion_dvars_correlation(ts, ts)
        np.testing.assert_allclose(corr, 1.0)

    def test_perfect_negative(self) -> None:
        """Perfectly anti-correlated → correlation = -1."""
        d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        f = np.array([0.0, 4.0, 3.0, 2.0, 1.0])
        corr = motion_dvars_correlation(d, f)
        np.testing.assert_allclose(corr, -1.0)

    def test_uncorrelated(self) -> None:
        """Orthogonal signals → correlation near 0."""
        rng = np.random.default_rng(10)
        n = 10000
        d = np.concatenate([[0.0], rng.standard_normal(n)])
        f = np.concatenate([[0.0], rng.standard_normal(n)])
        corr = motion_dvars_correlation(d, f)
        assert abs(corr) < 0.05

    def test_zero_variance_dvars(self) -> None:
        """Constant DVARS → correlation = 0 (undefined, fallback)."""
        d = np.array([0.0, 1.0, 1.0, 1.0])
        f = np.array([0.0, 0.1, 0.2, 0.3])
        corr = motion_dvars_correlation(d, f)
        assert corr == 0.0

    def test_zero_variance_fd(self) -> None:
        """Constant FD → correlation = 0 (undefined, fallback)."""
        d = np.array([0.0, 0.1, 0.2, 0.3])
        f = np.array([0.0, 0.5, 0.5, 0.5])
        corr = motion_dvars_correlation(d, f)
        assert corr == 0.0

    def test_single_volume_pair(self) -> None:
        """Only two volumes (one pair) → zero variance, returns 0."""
        d = np.array([0.0, 0.5])
        f = np.array([0.0, 0.3])
        corr = motion_dvars_correlation(d, f)
        assert corr == 0.0

    def test_skips_leading_zero(self) -> None:
        """Correlation is computed on [1:], ignoring leading zeros."""
        # With leading zeros the perfect correlation should still hold
        d = np.array([0.0, 2.0, 4.0, 6.0])
        f = np.array([0.0, 1.0, 2.0, 3.0])
        corr = motion_dvars_correlation(d, f)
        np.testing.assert_allclose(corr, 1.0)

    def test_returns_float(self) -> None:
        """Result should be a plain Python float."""
        d = np.array([0.0, 1.0, 2.0, 3.0])
        f = np.array([0.0, 3.0, 2.0, 1.0])
        corr = motion_dvars_correlation(d, f)
        assert isinstance(corr, float)


# ===================================================================
# dvars_qc_metrics
# ===================================================================
class TestDvarsQCMetrics:
    """Tests for the convenience wrapper returning all DVARS metrics."""

    def test_constant_signal(self) -> None:
        """Constant data → mean_dvars = 0, correlation = 0."""
        data = np.ones(SHAPE)
        fd = np.zeros(SHAPE[3])
        m = dvars_qc_metrics(data, _full_mask(), fd)
        assert m.mean_dvars == 0.0
        assert m.motion_dvars_corr == 0.0

    def test_returns_named_tuple(self) -> None:
        """Result is a DVARSQCMetrics named tuple with expected fields."""
        data = np.random.default_rng(20).standard_normal(SHAPE)
        fd = np.zeros(SHAPE[3])
        m = dvars_qc_metrics(data, _full_mask(), fd)
        assert isinstance(m, DVARSQCMetrics)
        assert hasattr(m, "mean_dvars")
        assert hasattr(m, "motion_dvars_corr")

    def test_integration(self) -> None:
        """Metrics are consistent with individual function outputs."""
        rng = np.random.default_rng(30)
        data = rng.standard_normal(SHAPE)
        fd = np.concatenate([[0.0], rng.random(SHAPE[3] - 1) * 0.3])

        m = dvars_qc_metrics(data, _full_mask(), fd)

        dv = dvars(data, _full_mask())
        expected_mean = float(np.mean(dv))
        expected_corr = motion_dvars_correlation(dv, fd)

        np.testing.assert_allclose(m.mean_dvars, expected_mean)
        np.testing.assert_allclose(m.motion_dvars_corr, expected_corr)

    def test_mean_dvars_nonnegative(self) -> None:
        """Mean DVARS should be non-negative."""
        data = np.random.default_rng(40).standard_normal(SHAPE)
        fd = np.zeros(SHAPE[3])
        m = dvars_qc_metrics(data, _full_mask(), fd)
        assert m.mean_dvars >= 0
