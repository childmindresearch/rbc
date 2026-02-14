"""Unit tests for rbc.core.qc.registration."""

from __future__ import annotations

import numpy as np

from rbc.core.qc.registration import (
    RegistrationQCMetrics,
    coverage,
    cross_correlation,
    dice_coefficient,
    jaccard_index,
    registration_qc_metrics,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
SHAPE = (10, 10, 10)


def _full_mask(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    return np.ones(shape, dtype=bool)


def _empty_mask(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def _half_mask(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    """First half of the x-axis is True."""
    m = np.zeros(shape, dtype=bool)
    m[: shape[0] // 2] = True
    return m


def _quarter_mask(shape: tuple[int, ...] = SHAPE) -> np.ndarray:
    """First quarter of the x-axis is True."""
    m = np.zeros(shape, dtype=bool)
    m[: shape[0] // 4] = True
    return m


# ===================================================================
# dice_coefficient
# ===================================================================
class TestDiceCoefficient:
    """Tests for Dice coefficient."""

    def test_identical_masks(self) -> None:
        """Identical non-empty masks → Dice = 1."""
        m = _full_mask()
        np.testing.assert_allclose(dice_coefficient(m, m), 1.0)

    def test_no_overlap(self) -> None:
        """Disjoint masks → Dice = 0."""
        m1 = _empty_mask()
        m1[:5] = True
        m2 = _empty_mask()
        m2[5:] = True
        assert dice_coefficient(m1, m2) == 0.0

    def test_both_empty(self) -> None:
        """Both empty → Dice = 0."""
        assert dice_coefficient(_empty_mask(), _empty_mask()) == 0.0

    def test_half_overlap(self) -> None:
        """Half-mask vs full-mask → Dice = 2 * 0.5 / 1.5 = 2/3."""
        full = _full_mask()
        half = _half_mask()
        expected = 2.0 * 500 / (1000 + 500)
        np.testing.assert_allclose(dice_coefficient(half, full), expected)

    def test_symmetry(self) -> None:
        """Dice(A, B) == Dice(B, A)."""
        rng = np.random.default_rng(0)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.3
        assert dice_coefficient(m1, m2) == dice_coefficient(m2, m1)

    def test_bounded(self) -> None:
        """Dice is in [0, 1]."""
        rng = np.random.default_rng(1)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.5
        d = dice_coefficient(m1, m2)
        assert 0.0 <= d <= 1.0

    def test_float_mask_treated_as_bool(self) -> None:
        """Float mask > 0 treated as True."""
        m = np.ones(SHAPE) * 0.7
        np.testing.assert_allclose(dice_coefficient(m, m), 1.0)


# ===================================================================
# jaccard_index
# ===================================================================
class TestJaccardIndex:
    """Tests for Jaccard index."""

    def test_identical_masks(self) -> None:
        """Identical non-empty masks → Jaccard = 1."""
        m = _full_mask()
        np.testing.assert_allclose(jaccard_index(m, m), 1.0)

    def test_no_overlap(self) -> None:
        """Disjoint masks → Jaccard = 0."""
        m1 = _empty_mask()
        m1[:5] = True
        m2 = _empty_mask()
        m2[5:] = True
        assert jaccard_index(m1, m2) == 0.0

    def test_both_empty(self) -> None:
        """Both empty → Jaccard = 0."""
        assert jaccard_index(_empty_mask(), _empty_mask()) == 0.0

    def test_half_overlap(self) -> None:
        """Half-mask vs full-mask → Jaccard = 500/1000 = 0.5."""
        full = _full_mask()
        half = _half_mask()
        np.testing.assert_allclose(jaccard_index(half, full), 0.5)

    def test_symmetry(self) -> None:
        """Jaccard(A, B) == Jaccard(B, A)."""
        rng = np.random.default_rng(2)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.3
        assert jaccard_index(m1, m2) == jaccard_index(m2, m1)

    def test_bounded(self) -> None:
        """Jaccard is in [0, 1]."""
        rng = np.random.default_rng(3)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.5
        j = jaccard_index(m1, m2)
        assert 0.0 <= j <= 1.0

    def test_dice_jaccard_relationship(self) -> None:
        """Dice = 2J / (1 + J)."""
        rng = np.random.default_rng(4)
        m1 = rng.random(SHAPE) > 0.4
        m2 = rng.random(SHAPE) > 0.6
        d = dice_coefficient(m1, m2)
        j = jaccard_index(m1, m2)
        np.testing.assert_allclose(d, 2 * j / (1 + j), atol=1e-12)


# ===================================================================
# cross_correlation
# ===================================================================
class TestCrossCorrelation:
    """Tests for Pearson correlation between binary masks."""

    def test_identical_masks(self) -> None:
        """Identical masks with variance → correlation = 1."""
        m = _half_mask()
        np.testing.assert_allclose(cross_correlation(m, m), 1.0)

    def test_complementary_masks(self) -> None:
        """Complementary masks → correlation = -1."""
        m1 = _half_mask()
        m2 = ~m1
        np.testing.assert_allclose(cross_correlation(m1, m2), -1.0)

    def test_both_full(self) -> None:
        """Both all-True → zero variance → 0."""
        assert cross_correlation(_full_mask(), _full_mask()) == 0.0

    def test_both_empty(self) -> None:
        """Both all-False → zero variance → 0."""
        assert cross_correlation(_empty_mask(), _empty_mask()) == 0.0

    def test_one_empty(self) -> None:
        """One all-False → zero variance → 0."""
        assert cross_correlation(_half_mask(), _empty_mask()) == 0.0

    def test_symmetry(self) -> None:
        """cross_correlation(A, B) == cross_correlation(B, A)."""
        rng = np.random.default_rng(5)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.3
        np.testing.assert_allclose(cross_correlation(m1, m2), cross_correlation(m2, m1))

    def test_bounded(self) -> None:
        """Correlation is in [-1, 1]."""
        rng = np.random.default_rng(6)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.5
        c = cross_correlation(m1, m2)
        assert -1.0 <= c <= 1.0


# ===================================================================
# coverage
# ===================================================================
class TestCoverage:
    """Tests for coverage index."""

    def test_identical_masks(self) -> None:
        """Identical non-empty masks → coverage = 1."""
        m = _full_mask()
        np.testing.assert_allclose(coverage(m, m), 1.0)

    def test_subset(self) -> None:
        """Smaller mask entirely within larger → coverage = 1."""
        full = _full_mask()
        quarter = _quarter_mask()
        np.testing.assert_allclose(coverage(quarter, full), 1.0)

    def test_no_overlap(self) -> None:
        """Disjoint masks → coverage = 0."""
        m1 = _empty_mask()
        m1[:5] = True
        m2 = _empty_mask()
        m2[5:] = True
        assert coverage(m1, m2) == 0.0

    def test_both_empty(self) -> None:
        """Both empty → coverage = 0."""
        assert coverage(_empty_mask(), _empty_mask()) == 0.0

    def test_one_empty(self) -> None:
        """One empty → coverage = 0."""
        assert coverage(_half_mask(), _empty_mask()) == 0.0

    def test_bounded(self) -> None:
        """Coverage is in [0, 1]."""
        rng = np.random.default_rng(7)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.5
        c = coverage(m1, m2)
        assert 0.0 <= c <= 1.0

    def test_symmetry(self) -> None:
        """coverage(A, B) == coverage(B, A)."""
        rng = np.random.default_rng(8)
        m1 = rng.random(SHAPE) > 0.5
        m2 = rng.random(SHAPE) > 0.3
        assert coverage(m1, m2) == coverage(m2, m1)


# ===================================================================
# registration_qc_metrics
# ===================================================================
class TestRegistrationQCMetrics:
    """Tests for the convenience wrapper returning all metrics."""

    def test_identical_masks(self) -> None:
        """Identical half-masks → dice=1, jaccard=1, corr=1, coverage=1."""
        m = _half_mask()
        r = registration_qc_metrics(m, m)
        np.testing.assert_allclose(r.dice, 1.0)
        np.testing.assert_allclose(r.jaccard, 1.0)
        np.testing.assert_allclose(r.cross_corr, 1.0)
        np.testing.assert_allclose(r.coverage, 1.0)

    def test_returns_named_tuple(self) -> None:
        """Result is a RegistrationQCMetrics with expected fields."""
        m = _full_mask()
        r = registration_qc_metrics(m, m)
        assert isinstance(r, RegistrationQCMetrics)
        assert hasattr(r, "dice")
        assert hasattr(r, "jaccard")
        assert hasattr(r, "cross_corr")
        assert hasattr(r, "coverage")

    def test_integration(self) -> None:
        """Metrics match individual function outputs."""
        rng = np.random.default_rng(10)
        m1 = rng.random(SHAPE) > 0.4
        m2 = rng.random(SHAPE) > 0.6
        r = registration_qc_metrics(m1, m2)
        np.testing.assert_allclose(r.dice, dice_coefficient(m1, m2))
        np.testing.assert_allclose(r.jaccard, jaccard_index(m1, m2))
        np.testing.assert_allclose(r.cross_corr, cross_correlation(m1, m2))
        np.testing.assert_allclose(r.coverage, coverage(m1, m2))

    def test_both_empty(self) -> None:
        """Both empty → all zeros."""
        r = registration_qc_metrics(_empty_mask(), _empty_mask())
        assert r.dice == 0.0
        assert r.jaccard == 0.0
        assert r.cross_corr == 0.0
        assert r.coverage == 0.0
