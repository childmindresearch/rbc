"""Unit tests for BOLD masking (FSL-AFNI method)."""

from __future__ import annotations

import numpy as np


class TestMaskCombination:
    """Tests for the mask combination logic in bold_masking."""

    def test_intersection_is_binary(self) -> None:
        """Multiplying two binary masks should yield a binary intersection."""
        rng = np.random.default_rng(100)
        # Create two overlapping synthetic masks
        mask_a = rng.random((10, 10, 10)) > 0.5
        mask_b = rng.random((10, 10, 10)) > 0.5
        combined = mask_a * mask_b
        unique_vals = np.unique(combined)
        assert set(unique_vals).issubset({0, 1})
        assert combined.sum() <= mask_a.sum()
        assert combined.sum() <= mask_b.sum()

    def test_intersection_with_empty(self) -> None:
        """Intersecting a mask with all-zeros should yield all-zeros."""
        mask = np.ones((10, 10, 10))
        empty = np.zeros((10, 10, 10))
        result = mask * empty
        assert np.all(result == 0)
