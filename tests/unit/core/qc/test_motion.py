"""Unit tests for rbc.core.qc.motion."""

from __future__ import annotations

import numpy as np

from rbc.core.qc.motion import (
    MotionQCMetrics,
    count_censored_volumes,
    framewise_displacement_jenkinson,
    framewise_displacement_power,
    motion_qc_metrics,
    rms_motion,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
N_VOLS = 100


def _zero_motion_params(n: int = N_VOLS) -> np.ndarray:
    """Return (n, 6) array of zeros (no motion)."""
    return np.zeros((n, 6))


# ===================================================================
# framewise_displacement_jenkinson
# ===================================================================
class TestFramewiseDisplacementJenkinson:
    """Tests for FD-Jenkinson from MCFLIRT relative RMS values."""

    def test_zero_motion(self) -> None:
        """Zero RMS values → FD all zeros."""
        rms = np.zeros(N_VOLS - 1)
        fd = framewise_displacement_jenkinson(rms)
        np.testing.assert_array_equal(fd, np.zeros(N_VOLS))

    def test_constant_motion(self) -> None:
        """Constant RMS → FD = [0, c, c, ...]."""
        rms = np.full(N_VOLS - 1, 0.15)
        fd = framewise_displacement_jenkinson(rms)
        assert fd[0] == 0.0
        np.testing.assert_allclose(fd[1:], 0.15)

    def test_known_values(self) -> None:
        """Specific input values produce expected output."""
        rms = np.array([0.1, 0.2, 0.3])
        fd = framewise_displacement_jenkinson(rms)
        expected = np.array([0.0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(fd, expected)

    def test_output_length(self) -> None:
        """Output should be one longer than input."""
        for n in [1, 5, 50, 200]:
            rms = np.zeros(n)
            fd = framewise_displacement_jenkinson(rms)
            assert len(fd) == n + 1

    def test_first_value_is_zero(self) -> None:
        """First FD value is always 0 (no predecessor)."""
        rms = np.random.default_rng(42).random(10) * 0.5
        fd = framewise_displacement_jenkinson(rms)
        assert fd[0] == 0.0


# ===================================================================
# framewise_displacement_power
# ===================================================================
class TestFramewiseDisplacementPower:
    """Tests for FD-Power from 6-column motion parameters."""

    def test_zero_motion(self) -> None:
        """Zero motion params → FD all zeros."""
        params = _zero_motion_params()
        fd = framewise_displacement_power(params)
        np.testing.assert_array_equal(fd, np.zeros(N_VOLS))

    def test_pure_translation(self) -> None:
        """Pure x-translation of 1mm/volume → FD = [0, 1, 1, ...]."""
        params = _zero_motion_params(5)
        params[:, 3] = np.arange(5, dtype=float)  # trans_x = 0, 1, 2, 3, 4
        fd = framewise_displacement_power(params)
        assert fd[0] == 0.0
        np.testing.assert_allclose(fd[1:], 1.0)

    def test_pure_rotation(self) -> None:
        """Pure rotation → FD scaled by 50mm sphere radius."""
        params = _zero_motion_params(3)
        # 1 radian rotation step in rot_x between vol 0 and vol 1
        params[1, 0] = 1.0
        fd = framewise_displacement_power(params)
        assert fd[0] == 0.0
        expected = 50.0 * np.pi / 180.0  # 50 * pi/180 * |1 rad|
        np.testing.assert_allclose(fd[1], expected)

    def test_known_values(self) -> None:
        """Hand-calculated example."""
        # 3 volumes: rot=(0,0,0), trans=(0,0,0) for all
        # Then set vol1 trans_x=1, vol2 trans_x=3
        params = np.zeros((3, 6))
        params[1, 3] = 1.0
        params[2, 3] = 3.0
        fd = framewise_displacement_power(params)
        # diff trans_x: [1, 2] → FD = [0, 1, 2]
        np.testing.assert_allclose(fd, [0.0, 1.0, 2.0])

    def test_output_length(self) -> None:
        """Output length matches number of volumes."""
        for n in [2, 10, 50]:
            params = _zero_motion_params(n)
            fd = framewise_displacement_power(params)
            assert len(fd) == n

    def test_first_value_is_zero(self) -> None:
        """First FD value is always 0."""
        rng = np.random.default_rng(7)
        params = rng.standard_normal((20, 6)) * 0.01
        fd = framewise_displacement_power(params)
        assert fd[0] == 0.0

    def test_nonnegative(self) -> None:
        """FD should always be non-negative."""
        rng = np.random.default_rng(8)
        params = rng.standard_normal((50, 6)) * 0.1
        fd = framewise_displacement_power(params)
        assert np.all(fd >= 0)


# ===================================================================
# rms_motion
# ===================================================================
class TestRmsMotion:
    """Tests for RMS of translation parameters."""

    def test_zero_motion(self) -> None:
        """Zero motion → (0, 0)."""
        params = _zero_motion_params()
        mean_rms, max_rms = rms_motion(params)
        assert mean_rms == 0.0
        assert max_rms == 0.0

    def test_known_values(self) -> None:
        """Single-volume with known translation → known RMS."""
        params = np.zeros((1, 6))
        params[0, 3:6] = [3.0, 4.0, 0.0]  # sqrt(9+16) = 5
        mean_rms, max_rms = rms_motion(params)
        np.testing.assert_allclose(mean_rms, 5.0)
        np.testing.assert_allclose(max_rms, 5.0)

    def test_only_translations_matter(self) -> None:
        """Rotations should not affect RMS motion."""
        params = np.zeros((5, 6))
        params[:, 0:3] = 100.0  # large rotations
        mean_rms, max_rms = rms_motion(params)
        assert mean_rms == 0.0
        assert max_rms == 0.0

    def test_max_greater_or_equal_mean(self) -> None:
        """Max RMS >= mean RMS always."""
        rng = np.random.default_rng(9)
        params = rng.standard_normal((50, 6))
        mean_rms, max_rms = rms_motion(params)
        assert max_rms >= mean_rms

    def test_multi_volume(self) -> None:
        """Two volumes with different translations."""
        params = np.zeros((2, 6))
        params[0, 3:6] = [1.0, 0.0, 0.0]  # RMS = 1
        params[1, 3:6] = [0.0, 3.0, 4.0]  # RMS = 5
        mean_rms, max_rms = rms_motion(params)
        np.testing.assert_allclose(mean_rms, 3.0)
        np.testing.assert_allclose(max_rms, 5.0)


# ===================================================================
# count_censored_volumes
# ===================================================================
class TestCountCensoredVolumes:
    """Tests for volume censoring based on FD threshold."""

    def test_all_below_threshold(self) -> None:
        """All FD values below threshold → 0 censored."""
        fd = np.full(N_VOLS, 0.1)
        assert count_censored_volumes(fd, fd_threshold=0.2) == 0

    def test_all_above_threshold(self) -> None:
        """All FD values above threshold → all censored."""
        fd = np.full(N_VOLS, 0.5)
        assert count_censored_volumes(fd, fd_threshold=0.2) == N_VOLS

    def test_mixed(self) -> None:
        """Some above, some below."""
        fd = np.array([0.0, 0.1, 0.3, 0.15, 0.5])
        assert count_censored_volumes(fd, fd_threshold=0.2) == 2

    def test_custom_threshold(self) -> None:
        """Custom threshold changes the count."""
        fd = np.array([0.0, 0.1, 0.3, 0.15, 0.5])
        assert count_censored_volumes(fd, fd_threshold=0.1) == 3
        assert count_censored_volumes(fd, fd_threshold=0.5) == 0

    def test_edge_at_threshold(self) -> None:
        """Exactly at threshold → NOT censored (strictly greater than)."""
        fd = np.array([0.2, 0.2, 0.2])
        assert count_censored_volumes(fd, fd_threshold=0.2) == 0

    def test_empty_array(self) -> None:
        """Empty FD array → 0 censored."""
        fd = np.array([])
        assert count_censored_volumes(fd) == 0


# ===================================================================
# motion_qc_metrics
# ===================================================================
class TestMotionQCMetrics:
    """Tests for the convenience wrapper returning all metrics."""

    def test_zero_motion(self) -> None:
        """Zero motion → all metrics zero."""
        rms_values = np.zeros(N_VOLS - 1)
        params = _zero_motion_params()
        m = motion_qc_metrics(rms_values, params)
        assert m.mean_fd == 0.0
        assert m.rel_means_rms_motion == 0.0
        assert m.rel_max_rms_motion == 0.0
        assert m.n_vol_censored == 0

    def test_returns_named_tuple(self) -> None:
        """Result is a MotionQCMetrics named tuple with expected fields."""
        rms_values = np.zeros(9)
        params = _zero_motion_params(10)
        m = motion_qc_metrics(rms_values, params)
        assert isinstance(m, MotionQCMetrics)
        assert hasattr(m, "mean_fd")
        assert hasattr(m, "rel_means_rms_motion")
        assert hasattr(m, "rel_max_rms_motion")
        assert hasattr(m, "n_vol_censored")

    def test_integration(self) -> None:
        """Metrics are consistent with individual function outputs."""
        rng = np.random.default_rng(99)
        rms_values = rng.random(N_VOLS - 1) * 0.3
        params = rng.standard_normal((N_VOLS, 6)) * 0.01

        m = motion_qc_metrics(rms_values, params, fd_threshold=0.15)

        fd = framewise_displacement_jenkinson(rms_values)
        expected_mean_fd = float(np.mean(fd))
        expected_mean_rms, expected_max_rms = rms_motion(params)
        expected_censored = count_censored_volumes(fd, 0.15)

        np.testing.assert_allclose(m.mean_fd, expected_mean_fd)
        np.testing.assert_allclose(m.rel_means_rms_motion, expected_mean_rms)
        np.testing.assert_allclose(m.rel_max_rms_motion, expected_max_rms)
        assert m.n_vol_censored == expected_censored

    def test_custom_threshold(self) -> None:
        """Custom FD threshold propagates correctly."""
        rms_values = np.array([0.1, 0.3, 0.5])
        params = _zero_motion_params(4)
        m_strict = motion_qc_metrics(rms_values, params, fd_threshold=0.05)
        m_lenient = motion_qc_metrics(rms_values, params, fd_threshold=0.4)
        assert m_strict.n_vol_censored > m_lenient.n_vol_censored
