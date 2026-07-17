"""Timeseries extraction and correlation matrix computation.

Extracts mean BOLD timeseries from atlas-defined ROIs and computes
pairwise Pearson correlation matrices.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

from rbc.core.niwrap import generate_exec_folder

logger = logging.getLogger(__name__)


def extract_timeseries(
    data: np.ndarray,
    atlas: np.ndarray,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Extract mean timeseries from atlas-defined ROIs.

    Args:
        data: 4D array (X, Y, Z, T) of BOLD data.
        atlas: 3D integer array (X, Y, Z) of ROI labels.  Label 0 is
            treated as background and excluded.
        labels: Optional 1D array of ROI labels to extract.  If *None*,
            all unique non-zero labels in the atlas are used (sorted).

    Returns:
        (n_rois, n_timepoints) float64 array of mean timeseries.

    Raises:
        ValueError: If *data* is not 4D, *atlas* is not 3D, spatial
            dimensions do not match, or no ROI labels are found.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got {data.ndim}D")
    if atlas.ndim != 3:
        raise ValueError(f"Expected 3D atlas, got {atlas.ndim}D")
    if data.shape[:3] != atlas.shape:
        raise ValueError(
            f"Spatial dimensions do not match: data {data.shape[:3]} "
            f"vs atlas {atlas.shape}"
        )

    atlas = atlas.astype(int)
    if labels is None:
        labels = np.unique(atlas)
        labels = labels[labels != 0]

    if len(labels) == 0:
        raise ValueError("No ROI labels found in atlas")

    n_timepoints = data.shape[3]
    timeseries = np.empty((len(labels), n_timepoints), dtype=np.float64)

    for idx, label in enumerate(labels):
        mask = atlas == label
        timeseries[idx] = data[mask].mean(axis=0)

    return timeseries


def correlation_matrix(timeseries: np.ndarray) -> np.ndarray:
    """Compute pairwise Pearson correlation matrix.

    ROIs with zero variance (constant signal) produce ``NaN`` entries
    because the Pearson correlation is undefined for constant series.
    A warning is logged for any such ROIs.

    Args:
        timeseries: (n_rois, n_timepoints) array.

    Returns:
        (n_rois, n_rois) symmetric correlation matrix.

    Raises:
        ValueError: If fewer than 2 ROIs or fewer than 2 timepoints.
    """
    if timeseries.shape[0] < 2:
        raise ValueError(f"Need at least 2 ROIs, got {timeseries.shape[0]}")
    if timeseries.shape[1] < 2:
        raise ValueError(f"Need at least 2 timepoints, got {timeseries.shape[1]}")

    n_rois = timeseries.shape[0]
    std = timeseries.std(axis=1)
    valid = std > 0

    if valid.all():
        return np.corrcoef(timeseries)

    n_zero = int((~valid).sum())
    logger.warning(
        "%d of %d ROIs have zero variance (constant signal); "
        "their correlation values will be NaN",
        n_zero,
        n_rois,
    )

    corr = np.full((n_rois, n_rois), np.nan, dtype=np.float64)
    np.fill_diagonal(corr, 1.0)
    if valid.sum() >= 2:
        corr[np.ix_(valid, valid)] = np.corrcoef(timeseries[valid])
    return corr


class TimeseriesOutputs(NamedTuple):
    """Outputs from :func:`compute_timeseries`."""

    timeseries: Path
    connectome: Path
    labels: np.ndarray


def compute_timeseries(
    in_file: str | Path,
    atlas_file: str | Path,
    out_dir: str | Path | None = None,
) -> TimeseriesOutputs:
    """Load NIfTI files, extract timeseries, and write results to disk.

    Thin wrapper around :func:`extract_timeseries` and
    :func:`correlation_matrix` that handles file I/O.

    Args:
        in_file: Path to 4D NIfTI (X, Y, Z, T).
        atlas_file: Path to 3D integer atlas NIfTI in the same space.
        out_dir: Output directory.  Defaults to the parent directory of
            *in_file*.

    Returns:
        :class:`TimeseriesOutputs` containing paths to the Parquet files and
        the ROI labels array.
    """
    import nibabel as nib
    from nibabel.processing import resample_from_to

    in_file = Path(in_file)
    img = nib.nifti1.load(in_file)
    atlas_img = nib.nifti1.load(atlas_file)

    # Resample atlas to BOLD grid if shapes differ (nearest-neighbor for labels)
    if atlas_img.shape[:3] != img.shape[:3]:
        atlas_img = resample_from_to(atlas_img, (img.shape[:3], img.affine), order=0)

    # Read via ``dataobj`` so the on-disk integer labels survive verbatim;
    # ``get_fdata`` would apply ``scl_slope``/``scl_inter`` and scale small
    # labels into garbage floats if the atlas ships with non-trivial scaling.
    atlas_data = np.asarray(atlas_img.dataobj).astype(int)

    data = img.get_fdata()
    labels = np.unique(atlas_data)
    labels = labels[labels != 0]

    ts = extract_timeseries(data, atlas_data, labels)
    corr = correlation_matrix(ts)

    if out_dir is None:
        out_dir = generate_exec_folder("timeseries")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = in_file.name.split(".nii")[0]
    ts_path = out_dir / f"{stem}_timeseries.parquet"
    corr_path = out_dir / f"{stem}_connectome.parquet"

    # ROIs
    roi_names = [str(label) for label in labels]
    # Timepoints
    tp_names = [str(i) for i in range(ts.shape[1])]

    pl.DataFrame(ts, schema=tp_names).write_parquet(ts_path)
    pl.DataFrame(corr, schema=roi_names).write_parquet(corr_path)

    return TimeseriesOutputs(
        timeseries=ts_path,
        connectome=corr_path,
        labels=labels,
    )
