"""Registration quality metrics.

Computes spatial overlap metrics between two binary masks to assess the
quality of coregistration (BOLD to T1w) and normalization (BOLD to
template).  All functions operate on plain NumPy boolean arrays (no
file I/O).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# Pass/fail threshold for longitudinal registration QC (Dice coefficient).
DICE_THRESHOLD = 0.85


class RegistrationQCMetrics(NamedTuple):
    """Spatial overlap metrics between two binary masks.

    Attributes:
        dice: Dice coefficient, ``2|A & B| / (|A| + |B|)``.
        jaccard: Jaccard index, ``|A & B| / |A | B|``.
        cross_corr: Pearson correlation between flattened masks.
        coverage: ``|A & B| / min(|A|, |B|)``.
    """

    dice: float
    jaccard: float
    cross_corr: float
    coverage: float


def dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the Dice coefficient between two binary masks.

    ``DC = 2|A & B| / (|A| + |B|)``

    Args:
        mask1: Boolean-compatible array.
        mask2: Boolean-compatible array of the same shape.

    Returns:
        Dice coefficient in ``[0, 1]``.  Returns ``0.0`` when both
        masks are empty.
    """
    m1 = np.asarray(mask1) > 0
    m2 = np.asarray(mask2) > 0
    size1 = int(np.count_nonzero(m1))
    size2 = int(np.count_nonzero(m2))
    if size1 + size2 == 0:
        return 0.0
    intersection = int(np.count_nonzero(m1 & m2))
    return 2.0 * intersection / (size1 + size2)


def jaccard_index(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the Jaccard index between two binary masks.

    ``JC = |A & B| / |A | B|``

    Args:
        mask1: Boolean-compatible array.
        mask2: Boolean-compatible array of the same shape.

    Returns:
        Jaccard index in ``[0, 1]``.  Returns ``0.0`` when both masks
        are empty.
    """
    m1 = np.asarray(mask1) > 0
    m2 = np.asarray(mask2) > 0
    union = int(np.count_nonzero(m1 | m2))
    if union == 0:
        return 0.0
    intersection = int(np.count_nonzero(m1 & m2))
    return intersection / union


def cross_correlation(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Pearson correlation between two flattened binary masks.

    Args:
        mask1: Boolean-compatible array.
        mask2: Boolean-compatible array of the same shape.

    Returns:
        Pearson *r* in ``[-1, 1]``.  Returns ``0.0`` when either mask
        has zero variance (all-True or all-False).
    """
    m1 = (np.asarray(mask1) > 0).ravel().astype(np.float64)
    m2 = (np.asarray(mask2) > 0).ravel().astype(np.float64)
    if np.std(m1) == 0 or np.std(m2) == 0:
        return 0.0
    return float(np.corrcoef(m1, m2)[0, 1])


def coverage(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the coverage index between two binary masks.

    ``coverage = |A & B| / min(|A|, |B|)``

    Measures how much of the smaller mask is covered by the overlap.

    Args:
        mask1: Boolean-compatible array.
        mask2: Boolean-compatible array of the same shape.

    Returns:
        Coverage in ``[0, 1]``.  Returns ``0.0`` when either mask is
        empty.
    """
    m1 = np.asarray(mask1) > 0
    m2 = np.asarray(mask2) > 0
    smaller = min(int(np.count_nonzero(m1)), int(np.count_nonzero(m2)))
    if smaller == 0:
        return 0.0
    intersection = int(np.count_nonzero(m1 & m2))
    return intersection / smaller


def registration_qc_metrics(
    mask1: np.ndarray,
    mask2: np.ndarray,
) -> RegistrationQCMetrics:
    """Compute all registration overlap metrics at once.

    Convenience wrapper that calls :func:`dice_coefficient`,
    :func:`jaccard_index`, :func:`cross_correlation`, and
    :func:`coverage`.

    Args:
        mask1: Boolean-compatible array (e.g. BOLD mask in target space).
        mask2: Boolean-compatible array (e.g. target brain mask).

    Returns:
        A :class:`RegistrationQCMetrics` named tuple.
    """
    return RegistrationQCMetrics(
        dice=dice_coefficient(mask1, mask2),
        jaccard=jaccard_index(mask1, mask2),
        cross_corr=cross_correlation(mask1, mask2),
        coverage=coverage(mask1, mask2),
    )
