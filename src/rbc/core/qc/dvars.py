"""DVARS and motion-DVARS correlation metrics.

DVARS (Derivative of RMS VARiance over voxelS) quantifies the rate of
change in BOLD signal intensity across the brain between consecutive
volumes.  Together with the correlation between DVARS and framewise
displacement it provides a complementary view of data quality beyond
pure head-motion estimates.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DVARSQCMetrics(NamedTuple):
    """Summary DVARS QC metrics for a single functional run.

    Attributes:
        mean_dvars: Mean DVARS value across volumes.
        motion_dvars_corr: Pearson correlation between DVARS and FD.
    """

    mean_dvars: float
    motion_dvars_corr: float


def dvars(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute DVARS from 4-D fMRI data.

    DVARS is defined as the root-mean-square of the temporal derivative
    of the BOLD signal across all in-mask voxels, giving one value per
    consecutive volume pair.  A leading zero is prepended so the output
    length matches the number of volumes.

    Args:
        data: 4-D array ``(X, Y, Z, T)``.
        mask: 3-D boolean-compatible brain mask.

    Returns:
        1-D float64 array of length *T* with ``dvars[0] = 0``.

    Raises:
        ValueError: If *data* is not 4-D.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")

    mask = np.asarray(mask) > 0
    diff = np.diff(data, axis=3)  # (X, Y, Z, T-1)
    masked = diff[mask]  # (n_voxels, T-1)
    dvars_values = np.sqrt(np.mean(masked**2, axis=0))
    return np.insert(dvars_values.astype(np.float64), 0, 0.0)


def motion_dvars_correlation(dvars_ts: np.ndarray, fd: np.ndarray) -> float:
    """Compute Pearson correlation between DVARS and framewise displacement.

    Both timeseries are expected to have a leading zero (for the first
    volume).  The correlation is computed on ``[1:]`` to exclude the
    trivial zero-zero pair.

    If either timeseries has zero variance after trimming, ``0.0`` is
    returned (correlation is undefined).

    Args:
        dvars_ts: 1-D DVARS timeseries (length *T*, first value 0).
        fd: 1-D framewise displacement timeseries (length *T*, first
            value 0).

    Returns:
        Pearson *r* between ``dvars_ts[1:]`` and ``fd[1:]``.
    """
    dvars_ts = np.asarray(dvars_ts, dtype=np.float64).ravel()
    fd = np.asarray(fd, dtype=np.float64).ravel()

    # Trim leading zero shared by both timeseries.
    d = dvars_ts[1:]
    f = fd[1:]

    if len(d) == 0 or np.std(d) == 0 or np.std(f) == 0:
        return 0.0

    return float(np.corrcoef(d, f)[0, 1])


def dvars_qc_metrics(
    data: np.ndarray,
    mask: np.ndarray,
    fd: np.ndarray,
) -> DVARSQCMetrics:
    """Compute DVARS and motion-DVARS correlation at once.

    Convenience wrapper that calls :func:`dvars` and
    :func:`motion_dvars_correlation`.

    Args:
        data: 4-D fMRI array ``(X, Y, Z, T)``.
        mask: 3-D boolean-compatible brain mask.
        fd: 1-D framewise displacement timeseries (length *T*).

    Returns:
        A :class:`DVARSQCMetrics` named tuple.
    """
    dvars_ts = dvars(data, mask)
    corr = motion_dvars_correlation(dvars_ts, fd)
    return DVARSQCMetrics(
        mean_dvars=float(np.mean(dvars_ts)),
        motion_dvars_corr=corr,
    )
