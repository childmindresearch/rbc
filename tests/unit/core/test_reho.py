"""Unit tests for rbc.core.metrics.reho."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import rankdata

from rbc.core.metrics.reho import (
    get_neighbor_offsets,
    kendall_w,
    rank_timeseries,
    reho,
)

if TYPE_CHECKING:
    from rbc.core.metrics.reho import ClusterSize


class TestKendallW:
    """Tests for Kendall's coefficient of concordance."""

    def test_perfect_concordance(self) -> None:
        """Identical timeseries should yield W=1."""
        ts = np.arange(20, dtype=np.float64)
        ranks = np.column_stack([rankdata(ts)] * 5)
        assert kendall_w(ranks) == pytest.approx(1.0)

    def test_zero_variance(self) -> None:
        """Constant ranks (all same value) should yield W=0."""
        ranks = np.ones((20, 5), dtype=np.float64)
        assert kendall_w(ranks) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """Check against a hand-computed example.

        3 judges, 4 subjects:
            Judge 1: [1, 2, 3, 4]
            Judge 2: [1, 3, 2, 4]
            Judge 3: [1, 2, 4, 3]
        Rank sums: [3, 7, 9, 11]  mean=7.5
        S = (3-7.5)^2 + (7-7.5)^2 + (9-7.5)^2 + (11-7.5)^2 = 35
        W = 12*35 / (9 * (64-4)) = 420 / 540 = 7/9
        """
        ranks = np.array(
            [
                [1, 1, 1],
                [2, 3, 2],
                [3, 2, 4],
                [4, 4, 3],
            ],
            dtype=np.float64,
        )
        assert kendall_w(ranks) == pytest.approx(7.0 / 9.0)

    def test_two_judges(self) -> None:
        """W with 2 judges should still be valid."""
        ranks = np.array(
            [
                [1, 2],
                [2, 1],
                [3, 3],
            ],
            dtype=np.float64,
        )
        result = kendall_w(ranks)
        assert 0.0 <= result <= 1.0


class TestNeighborOffsets:
    """Tests for 3D neighborhood offset generation."""

    @pytest.mark.parametrize(
        ("cluster_size", "expected_count"),
        [
            (7, 7),
            (19, 19),
            (27, 27),
        ],
    )
    def test_counts(self, cluster_size: ClusterSize, expected_count: int) -> None:
        """Offset array should have exactly cluster_size rows."""
        offsets = get_neighbor_offsets(cluster_size)
        assert offsets.shape == (expected_count, 3)

    def test_contain_center(self) -> None:
        """All neighbourhoods should include the center voxel."""
        for size in (7, 19, 27):
            offsets = get_neighbor_offsets(size)
            assert any(np.all(o == 0) for o in offsets)

    def test_7_is_subset_of_19(self) -> None:
        """Neighbourhood 7 should be a subset of 19."""
        off7 = set(map(tuple, get_neighbor_offsets(7)))
        off19 = set(map(tuple, get_neighbor_offsets(19)))
        assert off7.issubset(off19)

    def test_19_is_subset_of_27(self) -> None:
        """Neighbourhood 19 should be a subset of 27."""
        off19 = set(map(tuple, get_neighbor_offsets(19)))
        off27 = set(map(tuple, get_neighbor_offsets(27)))
        assert off19.issubset(off27)

    def test_7_face_only(self) -> None:
        """cluster_size=7 should only include face-adjacent + center (L1 <= 1)."""
        offsets = get_neighbor_offsets(7)
        for o in offsets:
            assert np.sum(np.abs(o)) <= 1

    def test_19_no_corners(self) -> None:
        """cluster_size=19 should exclude cube corners (all |di|=1)."""
        offsets = get_neighbor_offsets(19)
        for o in offsets:
            assert not np.all(np.abs(o) == 1)


class TestRankTimeseries:
    """Tests for voxelwise timeseries ranking."""

    def test_shape(self) -> None:
        """Output shape and dtype should match expectations."""
        data = np.random.default_rng(0).standard_normal((4, 5, 6, 10))
        ranks = rank_timeseries(data)
        assert ranks.shape == data.shape
        assert ranks.dtype == np.float32

    def test_range(self) -> None:
        """Ranks should be in [1, T] for each voxel."""
        nt = 20
        data = np.random.default_rng(1).standard_normal((3, 4, 5, nt))
        ranks = rank_timeseries(data)
        assert ranks.min() >= 1.0
        assert ranks.max() <= float(nt)

    def test_matches_scipy(self) -> None:
        """Spot-check a single voxel against scipy.stats.rankdata."""
        rng = np.random.default_rng(2)
        data = rng.standard_normal((3, 4, 5, 15))
        ranks = rank_timeseries(data)

        expected = rankdata(data[1, 2, 3, :])
        np.testing.assert_allclose(ranks[1, 2, 3, :], expected, atol=1e-5)

    def test_ties(self) -> None:
        """Tied values should get midranks."""
        data = np.zeros((2, 2, 2, 4))
        data[0, 0, 0, :] = [1.0, 1.0, 2.0, 3.0]
        ranks = rank_timeseries(data)
        np.testing.assert_allclose(ranks[0, 0, 0, :], [1.5, 1.5, 3.0, 4.0], atol=1e-5)


class TestReHo:
    """Tests for the full ReHo computation."""

    def test_output_shape(self) -> None:
        """Output shape should be 3D and match first 3 dims of data."""
        data = np.random.default_rng(3).standard_normal((8, 8, 8, 20))
        mask = np.ones((8, 8, 8))
        result = reho(data, mask, cluster_size=27)
        assert result.shape == (8, 8, 8)
        assert result.dtype == np.float64

    def test_respects_mask(self) -> None:
        """Voxels outside the mask should be zero."""
        rng = np.random.default_rng(4)
        data = rng.standard_normal((8, 8, 8, 20))
        mask = np.zeros((8, 8, 8))
        mask[2:6, 2:6, 2:6] = 1
        result = reho(data, mask, cluster_size=27)

        assert np.all(result[mask == 0] == 0.0)

    def test_identical_timeseries(self) -> None:
        """If all voxels have the same timeseries, W should be 1 everywhere."""
        ts = np.sin(np.linspace(0, 4 * np.pi, 30))
        data = np.tile(ts, (6, 6, 6, 1))
        mask = np.ones((6, 6, 6))
        result = reho(data, mask, cluster_size=7)

        interior = result[1:-1, 1:-1, 1:-1]
        nonzero = interior[interior > 0]
        np.testing.assert_allclose(nonzero, 1.0, atol=1e-10)

    def test_random_timeseries_bounded(self) -> None:
        """ReHo values should be in [0, 1]."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal((10, 10, 10, 25))
        mask = np.ones((10, 10, 10))
        result = reho(data, mask, cluster_size=27)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.parametrize("cluster_size", [7, 19, 27])
    def test_cluster_sizes(self, cluster_size: ClusterSize) -> None:
        """All cluster sizes should run without error and produce valid output."""
        rng = np.random.default_rng(6)
        data = rng.standard_normal((8, 8, 8, 15))
        mask = np.ones((8, 8, 8))
        result = reho(data, mask, cluster_size=cluster_size)

        assert result.shape == (8, 8, 8)
        assert np.all(np.isfinite(result))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_higher_concordance_for_correlated_neighbors(self) -> None:
        """Spatially correlated data should yield higher ReHo than noise."""
        rng = np.random.default_rng(7)
        nt = 30
        shape = (10, 10, 10)

        base_signal = rng.standard_normal(nt)
        correlated = np.empty((*shape, nt))
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    correlated[i, j, k, :] = base_signal + 0.1 * rng.standard_normal(nt)

        noise = rng.standard_normal((*shape, nt))
        mask = np.ones(shape)

        reho_corr = reho(correlated, mask, cluster_size=7)
        reho_noise = reho(noise, mask, cluster_size=7)

        assert np.mean(reho_corr) > np.mean(reho_noise)

    def test_empty_mask(self) -> None:
        """Empty mask should produce all-zero output."""
        data = np.random.default_rng(8).standard_normal((6, 6, 6, 10))
        mask = np.zeros((6, 6, 6))
        result = reho(data, mask, cluster_size=27)
        assert np.all(result == 0.0)

    def test_single_voxel_mask(self) -> None:
        """A single masked voxel has no neighbors, so it should be zero."""
        data = np.random.default_rng(9).standard_normal((6, 6, 6, 10))
        mask = np.zeros((6, 6, 6))
        mask[3, 3, 3] = 1
        result = reho(data, mask, cluster_size=27)
        assert result[3, 3, 3] == 0.0

    def test_rejects_non_4d(self) -> None:
        """ReHo should reject non-4D input."""
        with pytest.raises(ValueError, match="4D"):
            reho(np.zeros((5, 5, 5)), np.ones((5, 5, 5)), cluster_size=27)

    def test_boundary_voxels(self) -> None:
        """Edge voxels should still get computed if they have enough neighbors."""
        rng = np.random.default_rng(10)
        data = rng.standard_normal((6, 6, 6, 15))
        mask = np.ones((6, 6, 6))
        result = reho(data, mask, cluster_size=7)

        assert result[0, 3, 3] > 0.0
        assert result[3, 0, 3] > 0.0
        assert result[3, 3, 0] > 0.0

    def test_float_mask(self) -> None:
        """Mask with float values > 0 should be treated as True."""
        rng = np.random.default_rng(11)
        data = rng.standard_normal((6, 6, 6, 10))
        bool_mask = np.ones((6, 6, 6))
        float_mask = np.ones((6, 6, 6)) * 0.7

        result_bool = reho(data, bool_mask, cluster_size=7)
        result_float = reho(data, float_mask, cluster_size=7)

        np.testing.assert_array_equal(result_bool, result_float)

    def test_deterministic(self) -> None:
        """Same input should produce identical output."""
        rng = np.random.default_rng(12)
        data = rng.standard_normal((6, 6, 6, 10))
        mask = np.ones((6, 6, 6))

        r1 = reho(data, mask, cluster_size=19)
        r2 = reho(data, mask, cluster_size=19)

        np.testing.assert_array_equal(r1, r2)