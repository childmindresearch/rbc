"""Nuisance regressor computation (pure numpy).

Provides functions to compute derivatives, expand motion parameters to
24 columns, extract mean tissue signals, compute aCompCor components,
and assemble full regressor matrices for 36-parameter and aCompCor
nuisance regression.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_logger = logging.getLogger(__name__)

_MOTION_LABELS = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
_EXPAND_SUFFIXES = ["", "_deriv", "_sq", "_deriv_sq"]


def compute_derivative(signal: np.ndarray) -> np.ndarray:
    """Compute backward-difference derivative of a 1-D signal.

    Args:
        signal: 1-D array of length T.

    Returns:
        1-D array ``[0, signal[1]-signal[0], signal[2]-signal[1], ...]``.

    Raises:
        ValueError: If *signal* is not 1-D.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got {signal.ndim}D")
    deriv = np.empty_like(signal, dtype=np.float64)
    deriv[0] = 0.0
    deriv[1:] = np.diff(signal)
    return deriv


def expand_regressor(signal: np.ndarray) -> np.ndarray:
    """Expand a single regressor to 4 columns.

    Returns ``[original, derivative, squared, derivative_squared]``.

    Args:
        signal: 1-D array of length T.

    Returns:
        (T, 4) float64 array.

    Raises:
        ValueError: If *signal* is not 1-D.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got {signal.ndim}D")
    deriv = compute_derivative(signal)
    return np.column_stack([signal, deriv, signal**2, deriv**2]).astype(np.float64)


def expand_motion_params(params: np.ndarray) -> np.ndarray:
    """Expand 6 motion parameters to 24 columns.

    Each of the 6 parameters is expanded via :func:`expand_regressor`
    (original, derivative, squared, derivative-squared).

    Args:
        params: (T, 6) array of motion parameters.

    Returns:
        (T, 24) float64 array.

    Raises:
        ValueError: If *params* does not have exactly 6 columns.
    """
    if params.ndim != 2 or params.shape[1] != 6:
        raise ValueError(f"Expected (T, 6) motion parameters, got shape {params.shape}")
    expanded = [expand_regressor(params[:, i]) for i in range(6)]
    return np.hstack(expanded)


def extract_mean_signal(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract the mean timeseries from masked voxels.

    Args:
        data: 4-D array ``(X, Y, Z, T)``.
        mask: 3-D boolean-compatible mask.

    Returns:
        1-D float64 array of length T.

    Raises:
        ValueError: If *data* is not 4-D, *mask* is not 3-D, spatial
            dimensions do not match, or mask is empty.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D")
    if data.shape[:3] != mask.shape:
        raise ValueError(
            f"Spatial dimensions do not match: data {data.shape[:3]} "
            f"vs mask {mask.shape}"
        )
    mask_bool = mask > 0
    if not np.any(mask_bool):
        raise ValueError("Mask is empty")
    return data[mask_bool].mean(axis=0).astype(np.float64)


def compute_acompcor(
    data: np.ndarray,
    mask: np.ndarray,
    n_components: int = 5,
) -> np.ndarray:
    """Compute anatomical CompCor (aCompCor) components via DetrendPC.

    Matches C-PAC's ``calc_compcor_components``:

    1. Linear detrend each masked voxel timeseries
    2. Mean-center each voxel
    3. Z-score normalize each voxel (divide by std)
    4. SVD on the (T, n_voxels) matrix
    5. Return the first *n_components* left singular vectors

    Args:
        data: 4-D array ``(X, Y, Z, T)``.
        mask: 3-D boolean-compatible mask (union of CSF + WM).
        n_components: Number of principal components to retain.

    Returns:
        (T, n_components) float64 array.

    Raises:
        ValueError: If *data* is not 4-D, *mask* is not 3-D, spatial
            dimensions do not match, or there are too few voxels.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")
    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got {mask.ndim}D")
    if data.shape[:3] != mask.shape:
        raise ValueError(
            f"Spatial dimensions do not match: data {data.shape[:3]} "
            f"vs mask {mask.shape}"
        )

    mask_bool = mask > 0
    n_voxels = mask_bool.sum()
    if n_voxels < n_components:
        raise ValueError(f"Too few voxels ({n_voxels}) for {n_components} components")

    # Extract voxel timeseries: (n_voxels, T)
    voxel_ts = data[mask_bool].astype(np.float64)

    # 1. Linear detrend along temporal axis (matches scipy.signal.detrend)
    from scipy.signal import detrend

    voxel_ts = detrend(voxel_ts, axis=1, type="linear")

    # 2-3. Mean-center and z-score normalize each voxel (matches C-PAC)
    # Transpose to (T, n_voxels) for column-wise operations
    m = voxel_ts.T  # (T, n_voxels)
    m = m - m.mean(axis=0)
    std = m.std(axis=0)
    # Drop zero-variance voxels to avoid division by zero
    nonzero = std > 0
    m = m[:, nonzero]
    std = std[nonzero]
    m = m / std

    if m.shape[1] < n_components:
        raise ValueError(
            f"Too few non-zero-variance voxels ({m.shape[1]}) "
            f"for {n_components} components"
        )

    # 4-5. SVD and extract first n_components left singular vectors
    u, _s, _vt = np.linalg.svd(m, full_matrices=False)

    return u[:, :n_components].astype(np.float64)


def _motion_column_names() -> list[str]:
    """Return the 24 column names for expanded motion parameters."""
    return [
        f"{label}{suffix}" for label in _MOTION_LABELS for suffix in _EXPAND_SUFFIXES
    ]


def _tissue_column_names(tissues: list[str]) -> list[str]:
    """Return expanded column names for one or more tissue signals."""
    return [f"{tissue}{suffix}" for tissue in tissues for suffix in _EXPAND_SUFFIXES]


def assemble_36param_regressors(
    motion: np.ndarray,
    csf_signal: np.ndarray,
    wm_signal: np.ndarray,
    global_signal: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the 36-parameter regressor matrix.

    24 motion (6 params x 4 expansions) + 4 CSF + 4 WM + 4 global = 36.

    Args:
        motion: (T, 6) motion parameters.
        csf_signal: 1-D mean CSF timeseries.
        wm_signal: 1-D mean WM timeseries.
        global_signal: 1-D mean global (brain) timeseries.

    Returns:
        ``(matrix, column_names)`` where *matrix* is (T, 36) and
        *column_names* has 36 entries.
    """
    motion_24 = expand_motion_params(motion)
    csf_4 = expand_regressor(csf_signal)
    wm_4 = expand_regressor(wm_signal)
    global_4 = expand_regressor(global_signal)

    matrix = np.hstack([motion_24, csf_4, wm_4, global_4])
    names = _motion_column_names() + _tissue_column_names(["csf", "wm", "global"])
    return matrix, names


def assemble_acompcor_regressors(
    motion: np.ndarray,
    csf_signal: np.ndarray,
    wm_signal: np.ndarray,
    acompcor: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Assemble the aCompCor regressor matrix.

    24 motion + 5 aCompCor + 4 CSF + 4 WM = 37.

    Args:
        motion: (T, 6) motion parameters.
        csf_signal: 1-D mean CSF timeseries.
        wm_signal: 1-D mean WM timeseries.
        acompcor: (T, n_components) aCompCor components.

    Returns:
        ``(matrix, column_names)`` where *matrix* is (T, 37) and
        *column_names* has 37 entries.
    """
    motion_24 = expand_motion_params(motion)
    csf_4 = expand_regressor(csf_signal)
    wm_4 = expand_regressor(wm_signal)

    matrix = np.hstack([motion_24, acompcor, csf_4, wm_4])
    names = (
        _motion_column_names()
        + [f"acompcor_{i}" for i in range(acompcor.shape[1])]
        + _tissue_column_names(["csf", "wm"])
    )
    return matrix, names


def check_regressor_rank(
    regressors: np.ndarray,
    column_names: list[str],
) -> None:
    """Warn if the regressor matrix is rank-deficient.

    Logs a warning listing the near-zero singular values and the
    corresponding columns that are likely collinear.

    Args:
        regressors: (T, N) regressor matrix.
        column_names: List of N column names (for diagnostic messages).
    """
    _n_timepoints, n_regressors = regressors.shape
    rank = np.linalg.matrix_rank(regressors)
    if rank < n_regressors:
        _logger.warning(
            "Regressor matrix is rank-deficient: rank %d < %d columns. "
            "3dTproject may produce unreliable results. "
            "Columns: %s",
            rank,
            n_regressors,
            column_names,
        )


def write_regressor_file(
    regressors: np.ndarray,
    column_names: list[str],
    out_file: str | Path,
) -> Path:
    """Write a regressor matrix to an AFNI-compatible .1D file.

    The file has a comment header line with column names, followed by
    whitespace-delimited numeric data (one row per timepoint).

    Args:
        regressors: (T, N) regressor matrix.
        column_names: List of N column names.
        out_file: Output file path.

    Returns:
        Path to the written file.
    """
    out_file = Path(out_file)
    header = "# " + " ".join(column_names)
    np.savetxt(out_file, regressors, header=header, comments="", fmt="%.10f")
    return out_file
