"""Integration tests for registration quality metrics using real data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
import pytest

from rbc.core.qc.registration import (
    coverage,
    cross_correlation,
    dice_coefficient,
    jaccard_index,
    registration_qc_metrics,
)

if TYPE_CHECKING:
    from conftest import TestSubjectData


@pytest.mark.slow
def test_registration_qc_from_t1w(test_subject: TestSubjectData) -> None:
    """Derive brain masks from real T1w data and verify metric ranges."""
    t1w_img = nib.nifti1.load(test_subject.t1w)
    t1w_data = t1w_img.get_fdata()

    # Create two overlapping masks at different thresholds to simulate
    # a coregistration comparison (BOLD mask vs T1w mask).
    median_val = float(np.median(t1w_data[t1w_data > 0]))
    mask_tight = t1w_data > median_val
    mask_loose = t1w_data > (median_val * 0.5)

    # Both masks should have nonzero voxels
    assert np.count_nonzero(mask_tight) > 0
    assert np.count_nonzero(mask_loose) > 0

    # --- Individual metrics ---
    d = dice_coefficient(mask_tight, mask_loose)
    assert 0.0 < d <= 1.0

    j = jaccard_index(mask_tight, mask_loose)
    assert 0.0 < j <= 1.0

    cc = cross_correlation(mask_tight, mask_loose)
    assert -1.0 <= cc <= 1.0
    assert cc > 0  # overlapping masks should be positively correlated

    cov = coverage(mask_tight, mask_loose)
    assert 0.0 < cov <= 1.0

    # Tight is a subset of loose, so coverage should be high
    assert cov > 0.5

    # --- Combined metrics ---
    r = registration_qc_metrics(mask_tight, mask_loose)
    np.testing.assert_allclose(r.dice, d)
    np.testing.assert_allclose(r.jaccard, j)
    np.testing.assert_allclose(r.cross_corr, cc)
    np.testing.assert_allclose(r.coverage, cov)

    # Dice-Jaccard relationship
    np.testing.assert_allclose(r.dice, 2 * r.jaccard / (1 + r.jaccard), atol=1e-12)
