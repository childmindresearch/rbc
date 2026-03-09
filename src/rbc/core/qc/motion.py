"""Motion quality control metrics.

Computes framewise displacement (FD), RMS motion, and volume censoring
counts from MCFLIRT motion-correction outputs.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class MotionQCMetrics(NamedTuple):
    """Summary motion QC metrics for a single functional run.

    Attributes:
        mean_fd: Mean framewise displacement (Jenkinson), in mm.
        rel_means_rms_motion: Mean RMS of translation parameters.
        rel_max_rms_motion: Max RMS of translation parameters.
        n_vol_censored: Number of volumes exceeding the FD threshold.
    """

    mean_fd: float
    rel_means_rms_motion: float
    rel_max_rms_motion: float
    n_vol_censored: int


def framewise_displacement_jenkinson(rms_values: np.ndarray) -> np.ndarray:
    """Compute FD-Jenkinson from MCFLIRT relative RMS displacement values.

    Args:
        rms_values: 1-D array of relative RMS displacements from
            MCFLIRT ``_rel.rms`` (length N, one value per consecutive
            volume pair).

    Returns:
        1-D float64 array of length ``N + 1`` with 0 prepended for the
        first volume (which has no predecessor).
    """
    rms_values = np.asarray(rms_values, dtype=np.float64).ravel()
    return np.insert(rms_values, 0, 0.0)


def framewise_displacement_power(motion_params: np.ndarray) -> np.ndarray:
    """Compute FD-Power from 6-column motion parameters.

    Args:
        motion_params: ``(T, 6)`` array where columns are
            ``[roll, pitch, yaw, dS, dL, dP]``
            (degrees for rotations, mm for translations).

    Returns:
        1-D float64 array of length *T* with 0 for the first volume.
    """
    motion_params = np.asarray(motion_params, dtype=np.float64)
    params = motion_params.T  # (6, T)

    rotations = np.abs(np.diff(params[0:3, :], axis=1)).T  # (T-1, 3)
    translations = np.abs(np.diff(params[3:6, :], axis=1)).T  # (T-1, 3)

    fd = translations.sum(axis=1) + (50.0 * np.pi / 180.0) * rotations.sum(axis=1)
    return np.insert(fd, 0, 0.0)


def rms_motion(motion_params: np.ndarray) -> tuple[float, float]:
    """Compute RMS of translation parameters.

    Args:
        motion_params: ``(T, 6)`` array (same layout as
            :func:`framewise_displacement_power`).

    Returns:
        ``(mean_rms, max_rms)`` - the mean and maximum of the per-volume
        RMS translation magnitude.
    """
    motion_params = np.asarray(motion_params, dtype=np.float64)
    translations = motion_params[:, 3:6]
    rms = np.sqrt(np.sum(translations**2, axis=1))
    return float(np.mean(rms)), float(np.max(rms))


def count_censored_volumes(fd: np.ndarray, fd_threshold: float = 0.2) -> int:
    """Count volumes where framewise displacement exceeds a threshold.

    Args:
        fd: 1-D framewise displacement array.
        fd_threshold: Displacement threshold in mm (default 0.2).

    Returns:
        Number of volumes with ``FD > fd_threshold``.
    """
    fd = np.asarray(fd, dtype=np.float64)
    return int(np.sum(fd > fd_threshold))


def motion_qc_metrics(
    rms_values: np.ndarray,
    motion_params: np.ndarray,
    fd_threshold: float = 0.2,
) -> MotionQCMetrics:
    """Compute all motion QC metrics at once.

    Convenience wrapper that calls :func:`framewise_displacement_jenkinson`,
    :func:`rms_motion`, and :func:`count_censored_volumes`.

    Args:
        rms_values: Relative RMS values from MCFLIRT ``_rel.rms``.
        motion_params: ``(T, 6)`` motion parameter array.
        fd_threshold: FD censoring threshold in mm.

    Returns:
        A :class:`MotionQCMetrics` named tuple.
    """
    fd = framewise_displacement_jenkinson(rms_values)
    mean_rms, max_rms = rms_motion(motion_params)
    n_censored = count_censored_volumes(fd, fd_threshold)

    return MotionQCMetrics(
        mean_fd=float(np.mean(fd)),
        rel_means_rms_motion=mean_rms,
        rel_max_rms_motion=max_rms,
        n_vol_censored=n_censored,
    )
